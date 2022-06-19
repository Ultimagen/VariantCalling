import os

import requests
from google.auth import default
from google.auth.transport.requests import Request

GOOGLE_APPLICATION_CREDENTIALS = "GOOGLE_APPLICATION_CREDENTIALS"
GCS_OAUTH_TOKEN = "GCS_OAUTH_TOKEN"
SCOPE = "https://www.googleapis.com/auth/cloud-platform"
URL = "https://www.googleapis.com/oauth2/v1/tokeninfo"
PAYLOAD = "access_token={}"
HEADERS = {"content-type": "application/x-www-form-urlencoded"}
OK = "OK"


def get_gcs_token():
    """
    Returns gcs access token.
    Ideally, this function generates a new token, requries that GOOGLE_APPLICATION_CREDENTIALS be set in the environment
    (os.environ).
    Alternatively, environment variable GCS_OAUTH_TOKEN could be set if a token already exists
    """
    if GOOGLE_APPLICATION_CREDENTIALS in os.environ:
        # getting the credentials and project details for gcp project
        # (second item is the project id, but it is not used later on)
        credentials, _ = default(scopes=[SCOPE])

        # getting request object
        auth_req = Request()
        credentials.refresh(auth_req)  # refresh token
        token = credentials.token
    elif GCS_OAUTH_TOKEN in os.environ:
        token = os.environ[GCS_OAUTH_TOKEN]
    else:
        raise ValueError(
            f"""Could not generate gcs token because {GOOGLE_APPLICATION_CREDENTIALS} is not set in the environment.
Alternatively, environment variable {GCS_OAUTH_TOKEN} could be set if a token already exists, but it was not"""
        )

    req = requests.post(URL, data=PAYLOAD.format(token), headers=HEADERS)
    if not req.reason == OK:
        raise ValueError(f"Could not verify token {token}\n\nResponse from server:\n{req.text}")
    if not req.json()["expires_in"] > 0:
        raise ValueError(f"token {token} expired")
    return token
