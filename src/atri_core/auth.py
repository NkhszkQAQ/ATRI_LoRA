"""Existing per-user bearer validation shared by native transports."""
import hashlib
import hmac

class MemoryService:
    def __init__(self, store, users):
        if (not isinstance(users, dict) or not users
                or any(not isinstance(k, str) or not k.strip() or not isinstance(v, str)
                       or len(v) < 32 for k, v in users.items())
                or len(set(users.values())) != len(users)):
            raise ValueError("memory users must map unique user IDs to unique tokens of at least 32 characters")
        self.store = store
        self.users = {hashlib.sha256(token.encode()).digest(): user for user, token in users.items()}

    def owner(self, authorization):
        from fastapi import HTTPException
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() == "bearer" and token:
            digest = hashlib.sha256(token.encode()).digest()
            for known, user in self.users.items():
                if hmac.compare_digest(digest, known):
                    return user
        raise HTTPException(401, "valid per-user Bearer token required", headers={"WWW-Authenticate": "Bearer"})
