import hashlib
import re
import secrets
from datetime import datetime, timedelta

import bcrypt
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import Config
from backend.db.models.refresh_token_model import RefreshTokenModel
from backend.db.models.settings_model import SettingModel
from backend.db.models.user_model import UserModel
from backend.db.session import get_async_session
from backend.enums.settings_enum import ESettingKey
from backend.enums.user_role_enum import EUserRole
from backend.helpers.now import now
from backend.schemas.auth_schema import TokenResponseSchema

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

# bcrypt only ever looks at the first 72 bytes of a password. Up to 4.x the
# library cut longer input silently; from 5.0 it raises ValueError instead.
BCRYPT_MAX_PASSWORD_BYTES = 72


def verify_password(plain: str, hashed: str) -> bool:
    # Cut to what bcrypt considers, rather than letting 5.x raise: a login
    # with a long password would otherwise be a 500, and an account whose
    # long password was hashed by an older bcrypt (which truncated it) could
    # no longer log in at all. The hash was built from these same 72 bytes.
    plain_bytes = plain.encode("utf-8")[:BCRYPT_MAX_PASSWORD_BYTES]
    hashed_bytes = hashed.encode("utf-8")
    return bcrypt.checkpw(plain_bytes, hashed_bytes)


def get_password_hash(password: str) -> str:
    pwd_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password=pwd_bytes, salt=salt)
    return hashed_password.decode("utf-8")


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = now() + (
        expires_delta or timedelta(minutes=Config.ACCESS_TOKEN_LIFETIME_MIN)
    )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode,
        Config.JWT_SECRET_KEY,
        algorithm=Config.JWT_ALGORITHM,
    )
    return encoded_jwt, expire


def access_token_claims(user: UserModel) -> dict:
    """What an access token says about its user.

    The subject is the user id, which never changes. It used to be the
    username, which the user can change: a token issued before a rename
    then pointed to whoever took that name next. `uid` marks the new form,
    so a legacy token whose username happens to look like a number is never
    read as an id.
    """
    return {"sub": str(user.id), "uid": user.id}


def hash_refresh_token(token: str) -> str:
    """What refresh_tokens.token stores: a sha256, so a copy of the database
    (a backup, a leaked volume) holds no session anyone can use. A fast hash
    is enough for 256 random bits; there is nothing to brute-force."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def refresh_token_lookup(token: str) -> list[str]:
    """The stored values that match a refresh token sent by a client.

    Its hash, and for one refresh-token lifetime after the upgrade, the token
    itself: rows written before it hold the raw value. A raw token is 43
    URL-safe characters, never 64 hex digits, so presenting a stored hash
    as if it were a token matches nothing.
    """
    hashed = hash_refresh_token(token)
    return [hashed] if _SHA256_HEX.fullmatch(token) else [hashed, token]


async def create_refresh_token(
    user_id: int,
    session: AsyncSession,
    user_agent: str | None = None,
    ip: str | None = None,
) -> str:
    """Stores a new refresh token (hashed) and returns it, in the clear."""
    token = secrets.token_urlsafe(32)
    expires_at = now() + timedelta(minutes=Config.REFRESH_TOKEN_LIFETIME_MIN)
    db_token = RefreshTokenModel(
        token=hash_refresh_token(token),
        user_id=user_id,
        expires_at=expires_at,
        user_agent=user_agent,
        ip_address=ip,
    )
    session.add(db_token)
    await session.commit()
    return token


async def authenticate_user(session: AsyncSession, username: str, password: str):
    stmt = select(UserModel).where(UserModel.username == username).limit(1)
    result = await session.execute(stmt)
    user = result.scalar_one_or_none()
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user


def build_token_response(
    access_token: str,
    access_exp: datetime,
    refresh_token: str,
):
    return TokenResponseSchema(
        access_token=access_token,
        token_type="bearer",
        expires_in=int((access_exp - now()).total_seconds()),
        refresh_token=refresh_token,
    )


async def is_user(
    token: str = Depends(oauth2_scheme),
    session: AsyncSession = Depends(get_async_session),
) -> UserModel:
    """
    Check if the request credentials is valid.
    Returns user.
    Raise 401 on fail
    """
    credentials_exception = HTTPException(status_code=401, detail="Invalid token")
    try:
        payload = jwt.decode(
            token,
            Config.JWT_SECRET_KEY,
            algorithms=[Config.JWT_ALGORITHM],
        )
        subject = payload.get("sub")
        uid = payload.get("uid")
        if not subject:
            raise credentials_exception
    except JWTError:
        raise credentials_exception from None
    if uid is not None:
        if not isinstance(uid, int) or subject != str(uid):
            raise credentials_exception
        stmt = select(UserModel).where(UserModel.id == uid).limit(1)
    else:
        # Legacy token (subject = username), still in circulation for at most
        # ACCESS_TOKEN_LIFETIME_MIN after the upgrade. Remove in the next release.
        stmt = select(UserModel).where(UserModel.username == subject).limit(1)
    result = await session.execute(stmt)
    user = result.scalar_one_or_none()
    if not user:
        raise credentials_exception
    return user


async def is_user_admin(
    token: str = Depends(oauth2_scheme),
    session: AsyncSession = Depends(get_async_session),
) -> UserModel:
    """
    Check if the request credentials is valid and match admin/owner.
    Returns user.
    Raise 403 on fail.
    """
    user = await is_user(token, session)
    roles: list[EUserRole] = [EUserRole.OWNER, EUserRole.ADMIN]
    if user.role_code in roles:
        return user
    raise HTTPException(403, "Not admin user")


async def is_user_owner(
    token: str = Depends(oauth2_scheme),
    session: AsyncSession = Depends(get_async_session),
) -> UserModel:
    """
    Check if the request credentials is valid and match owner.
    Returns user.
    Raise 403 on fail.
    """
    user = await is_user(token, session)
    if user.role_code == EUserRole.OWNER:
        return user
    raise HTTPException(403, "Not owner user")


async def allow_registration(
    session: AsyncSession = Depends(get_async_session),
):
    """Check if user registration is allowed. Raise  403 on fail."""
    stmt = (
        select(SettingModel)
        .where(SettingModel.key == ESettingKey.ALLOW_REGISTRATION)
        .limit(1)
    )
    result = await session.execute(stmt)
    setting = result.scalar_one_or_none()
    if not setting or setting.value.lower() != "true":
        raise HTTPException(403, "registration feature disabled in app settings")
    return True


async def is_creds_taken(
    session: AsyncSession,
    username: str,
    email: str,
    user_id: int | None,
):
    """
    Check if user credentials already taken by
    another user.
    """
    stmt = (
        select(UserModel)
        .where(
            or_(
                UserModel.username == username,
                UserModel.email == email,
            ),
            UserModel.id != user_id,
        )
        .limit(1)
    )
    result = await session.execute(stmt)
    user = result.scalar_one_or_none()
    return user is not None
