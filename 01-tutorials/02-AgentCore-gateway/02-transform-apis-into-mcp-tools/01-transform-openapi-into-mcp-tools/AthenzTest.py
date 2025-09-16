import base64, json, time
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")

# JWT header and payload
header = {
    "alg": "ES256",
    "typ": "JWT",
    "kid": "v0"
}
payload = {
    "iss": "idb2b.finance.agentic-test.217f928e-f18b-4123-9682-dd320fc1fcb4",
    "sub": "idb2b.finance.agentic-test.217f928e-f18b-4123-9682-dd320fc1fcb4",
    "aud": "https://id-uat.b2b.yahooincapis.com/zts/v1", 
    "exp": int(time.time()) + 10 * 60 * 60  # 10 minutes
}

# Encode header and payload
encoded_header = b64url(json.dumps(header, separators=(",", ":")).encode())
encoded_payload = b64url(json.dumps(payload, separators=(",", ":")).encode())
signing_input = f"{encoded_header}.{encoded_payload}".encode()

# Load EC private key
with open("./client_private_key.pem", "rb") as key_file:
    private_key = serialization.load_pem_private_key(key_file.read(), password=None)

# Sign and convert DER → raw (r||s)
der_signature = private_key.sign(signing_input, ec.ECDSA(hashes.SHA256()))
r, s = decode_dss_signature(der_signature)
r_bytes = r.to_bytes(32, byteorder="big")
s_bytes = s.to_bytes(32, byteorder="big")
raw_signature = r_bytes + s_bytes

# Encode raw signature to base64url
encoded_signature = b64url(raw_signature)

# Final JWT
jwt_token = f"{encoded_header}.{encoded_payload}.{encoded_signature}"
print("JWT Client Assertion:\n", jwt_token)
