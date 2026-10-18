"""Access tokens name their user by id, not by the (changeable) username."""

import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.core.auth_core import access_token_claims, create_access_token, is_user
from backend.db.models import BaseModel, UserModel


@pytest_asyncio.fixture
async def session(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(BaseModel.metadata.create_all)
    maker = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with maker() as db:
        yield db
    await engine.dispose()


async def _user(db, username: str, email: str | None = None) -> UserModel:
    user = UserModel(
        username=username, email=email or f"{username}@example.com", hashed_password="x"
    )
    db.add(user)
    await db.commit()
    return user


@pytest.mark.asyncio
async def test_a_token_follows_its_user_through_a_rename(session) -> None:
    alice = await _user(session, "alice")
    token, _ = create_access_token(access_token_claims(alice))

    alice.username = "alice2"
    await session.commit()
    # Someone else takes the old name: the token must not switch to them.
    await _user(session, "alice", email="impostor@example.com")

    assert (await is_user(token, session)).id == alice.id


@pytest.mark.asyncio
async def test_a_legacy_username_token_still_works_for_one_release(session) -> None:
    bob = await _user(session, "bob")
    token, _ = create_access_token({"sub": "bob"})

    assert (await is_user(token, session)).id == bob.id


@pytest.mark.asyncio
async def test_a_numeric_username_in_a_legacy_token_is_not_read_as_an_id(
    session,
) -> None:
    first = await _user(session, "first")
    numeric = await _user(session, str(first.id))  # username "1"
    token, _ = create_access_token({"sub": str(first.id)})

    assert (await is_user(token, session)).id == numeric.id


@pytest.mark.asyncio
async def test_a_token_whose_uid_and_subject_disagree_is_refused(session) -> None:
    carol = await _user(session, "carol")
    token, _ = create_access_token({"sub": "999", "uid": carol.id})

    with pytest.raises(HTTPException) as e:
        await is_user(token, session)
    assert e.value.status_code == 401


@pytest.mark.asyncio
async def test_a_token_for_a_deleted_user_is_refused(session) -> None:
    dave = await _user(session, "dave")
    token, _ = create_access_token(access_token_claims(dave))
    await session.delete(dave)
    await session.commit()

    with pytest.raises(HTTPException) as e:
        await is_user(token, session)
    assert e.value.status_code == 401
