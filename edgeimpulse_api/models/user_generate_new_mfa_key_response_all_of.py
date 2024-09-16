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



from pydantic import BaseModel, Field, StrictStr

class UserGenerateNewMfaKeyResponseAllOf(BaseModel):
    key: StrictStr = Field(..., description="Secret key (use SHA-1).")
    url: StrictStr = Field(..., description="URL that will be converted into a QR code that can be scanned.")
    __properties = ["key", "url"]

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
    def from_json(cls, json_str: str) -> UserGenerateNewMfaKeyResponseAllOf:
        """Create an instance of UserGenerateNewMfaKeyResponseAllOf from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> UserGenerateNewMfaKeyResponseAllOf:
        """Create an instance of UserGenerateNewMfaKeyResponseAllOf from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return UserGenerateNewMfaKeyResponseAllOf.construct(**obj)

        _obj = UserGenerateNewMfaKeyResponseAllOf.construct(**{
            "key": obj.get("key"),
            "url": obj.get("url")
        })
        return _obj

