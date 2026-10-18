"""Refresh tokens are stored as a sha256: a copy of the database opens no session."""

from datetime import timedelta

import pytest
import pytest_asyncio
from fastapi import HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.api.auth_api import refresh_token
from backend.core.auth_core import create_refresh_token, hash_refresh_token
from backend.db.models import BaseModel, RefreshTokenModel, UserModel
from backend.helpers.now import now
from backend.schemas.auth_schema import RefreshRequestSchema

REQUEST = Request({"type": "http", "headers": [], "client": None})


@pytest_asyncio.fixture
async def session(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(BaseModel.metadata.create_all)
    maker = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with maker() as db:
        yield db
    await engine.dispose()


async def _user(db) -> UserModel:
    user = UserModel(username="erin", email="erin@example.com", hashed_password="x")
    db.add(user)
    await db.commit()
    return user


async def _refresh(db, token: str):
    return await refresh_token(REQUEST, RefreshRequestSchema(refresh_token=token), db)


@pytest.mark.asyncio
async def test_the_database_holds_the_hash_and_the_client_the_token(session) -> None:
    user = await _user(session)
    token = await create_refresh_token(user.id, session)

    stored = (await session.execute(select(RefreshTokenModel.token))).scalar_one()
    assert stored == hash_refresh_token(token)
    assert stored != token
    assert (await _refresh(session, token)).refresh_token != token  # rotated


@pytest.mark.asyncio
async def test_the_stored_hash_is_not_a_usable_token(session) -> None:
    user = await _user(session)
    token = await create_refresh_token(user.id, session)

    with pytest.raises(HTTPException) as e:
        await _refresh(session, hash_refresh_token(token))
    assert e.value.status_code == 401


@pytest.mark.asyncio
async def test_a_token_stored_in_the_clear_before_the_upgrade_still_works(
    session,
) -> None:
    user = await _user(session)
    session.add(
        RefreshTokenModel(
            token="legacy-raw-token",
            user_id=user.id,
            expires_at=now() + timedelta(days=1),
        )
    )
    await session.commit()

    response = await _refresh(session, "legacy-raw-token")
    stored = (
        await session.execute(
            select(RefreshTokenModel.token).where(RefreshTokenModel.revoked.is_(False))
        )
    ).scalar_one()
    assert stored == hash_refresh_token(response.refresh_token)  # the new one is hashed
