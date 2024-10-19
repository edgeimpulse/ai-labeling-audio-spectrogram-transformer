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


from typing import List
from pydantic import BaseModel, Field, StrictStr

class UserSetTotpMfaKeyResponseAllOf(BaseModel):
    recovery_codes: List[StrictStr] = Field(..., alias="recoveryCodes", description="10 recovery codes, which can be used in case you've lost access to your MFA TOTP app. Recovery codes are single use. Once you've used a recovery code once, it can not be used again.")
    __properties = ["recoveryCodes"]

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
    def from_json(cls, json_str: str) -> UserSetTotpMfaKeyResponseAllOf:
        """Create an instance of UserSetTotpMfaKeyResponseAllOf from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> UserSetTotpMfaKeyResponseAllOf:
        """Create an instance of UserSetTotpMfaKeyResponseAllOf from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return UserSetTotpMfaKeyResponseAllOf.construct(**obj)

        _obj = UserSetTotpMfaKeyResponseAllOf.construct(**{
            "recovery_codes": obj.get("recoveryCodes")
        })
        return _obj

