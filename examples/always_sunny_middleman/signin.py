# type: ignore
# Runs in Middleman's image, for the signing library Middleman itself uses;
# this repository doesn't install it.
"""A local sign-in for the example stack: one key, one token, and the key served for Middleman to check.

A new key and token are made on each start, so a token from an earlier run stops working. The token lets its holder use the models in `middleman/models.jsonc` through this stack and nothing else.
"""

import http.server
import json
import time
from pathlib import Path

from joserfc import jwt
from joserfc.jwk import KeySet, RSAKey

ISSUER = "http://signin:18500/"
AUDIENCE = "always-sunny"
PORT = 18500
VALID_FOR = 7 * 24 * 60 * 60

key = RSAKey.generate_key(
    2048, parameters={"kid": "local-1", "use": "sig", "alg": "RS256"}
)
Path("jwks.json").write_text(json.dumps(KeySet([key]).as_dict(private=False)))

now = int(time.time())
claims = {
    "iss": ISSUER,
    "aud": AUDIENCE,
    "sub": "local-dev",
    "permissions": ["model-access-public"],
    "iat": now,
    "exp": now + VALID_FOR,
}
Path("token.txt").write_text(
    jwt.encode({"alg": "RS256", "kid": "local-1"}, claims, key)
)

http.server.ThreadingHTTPServer(
    ("0.0.0.0", PORT), http.server.SimpleHTTPRequestHandler
).serve_forever()
