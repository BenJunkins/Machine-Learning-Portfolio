"""

A web API for the spam filter: send an email / predict and get back how likely it is spam

"""

from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, StringConstraints

from spam_filter.predict import label_for, load_model, spam_probability

MAX_EMAIL_CHARACTERS = 100_000


EXAMPLE_EMAIL = (
    "Congratulations! You have been selected to receive a FREE $1000 gift card. "
    "Click here to claim your prize: http://www.example.com/claim"
)


class EmailIn(BaseModel):
    email: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True, min_length=1, max_length=MAX_EMAIL_CHARACTERS
        ),
        Field(
            description="Just the body text, or the full raw email with headers",
            examples=[EXAMPLE_EMAIL],
        ),
    ]


class PredictionOut(BaseModel):
    spam_probability: float = Field(
        ge=0, le=1, description="Share of trees in the forest that voted spam"
    )
    label: Literal["ham", "unsure", "spam"] = Field(
        description="ham below 0.4, spam at 0.6 and above, unsure in between"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()  # Runs once when the server starts
    yield


app = FastAPI(
    title="Spam Filter API",
    summary="Scores an email with a TF-IDF and Random Forest model trained on SpamAssassin.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(body: EmailIn, request: Request) -> PredictionOut:
    probability = float(spam_probability(request.app.state.model, body.email))

    return PredictionOut(
        spam_probability=round(probability, 4), label=label_for(probability)
    )
