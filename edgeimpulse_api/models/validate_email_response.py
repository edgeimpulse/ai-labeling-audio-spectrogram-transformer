# coding: utf-8

"""
    Edge Impulse API

    No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)  # noqa: E501

    The version of the OpenAPI document: 1.0.0
    Generated by: https://openapi-generator.tech
"""


from __future__ import annotations
from inspect import getfullargspec
import pprint
import re  # noqa: F401
import json


from typing import Optional
from pydantic import BaseModel, Field, StrictBool, StrictStr, validator

class ValidateEmailResponse(BaseModel):
    success: StrictBool = Field(..., description="Whether the operation succeeded")
    error: Optional[StrictStr] = Field(None, description="Optional error description (set if 'success' was false)")
    email: StrictStr = Field(..., description="Email address that was checked.")
    verdict: StrictStr = Field(..., description="Classification of the email's validity status")
    score: float = Field(..., description="This number from 0 to 1 represents the likelihood the email address is valid, expressed as a percentage.")
    suggestion: Optional[StrictStr] = Field(None, description="A corrected domain, if a possible typo is detected.")
    local: Optional[StrictStr] = Field(None, description="The first part of the email address (before the @ sign)")
    host: Optional[StrictStr] = Field(None, description="The second part of the email address (after the @ sign)")
    __properties = ["success", "error", "email", "verdict", "score", "suggestion", "local", "host"]

    @validator('verdict')
    def verdict_validate_enum(cls, v):
        if v not in ('Valid', 'Risky', 'Invalid'):
            raise ValueError("must validate the enum values ('Valid', 'Risky', 'Invalid')")
        return v

    class Config:
        allow_population_by_field_name = True
        validate_assignment = False

    def to_str(self) -> str:
        """Returns the string representation of the model using alias"""
        return pprint.pformat(self.dict(by_alias=True))

    def to_json(self) -> str:
        """Returns the JSON representation of the model using alias"""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> ValidateEmailResponse:
        """Create an instance of ValidateEmailResponse from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> ValidateEmailResponse:
        """Create an instance of ValidateEmailResponse from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return ValidateEmailResponse.construct(**obj)

        _obj = ValidateEmailResponse.construct(**{
            "success": obj.get("success"),
            "error": obj.get("error"),
            "email": obj.get("email"),
            "verdict": obj.get("verdict"),
            "score": obj.get("score"),
            "suggestion": obj.get("suggestion"),
            "local": obj.get("local"),
            "host": obj.get("host")
        })
        return _obj

