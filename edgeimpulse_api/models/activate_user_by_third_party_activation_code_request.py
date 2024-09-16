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
from pydantic import BaseModel, Field, StrictBool, StrictStr

class ActivateUserByThirdPartyActivationCodeRequest(BaseModel):
    activation_code: StrictStr = Field(..., alias="activationCode")
    password: StrictStr = Field(..., description="Password, minimum length 8 characters.")
    name: Optional[StrictStr] = Field(None, description="Your name")
    username: StrictStr = Field(..., description="Username, minimum 4 and maximum 30 characters. May contain alphanumeric characters, hyphens, underscores and dots. Validated according to `^(?=.{4,30}$)(?![_.])(?!.*[_.]{2})[a-zA-Z0-9._-]+(?<![_.])$`.")
    privacy_policy: Optional[StrictBool] = Field(None, alias="privacyPolicy", description="Whether the user accepted the privacy policy")
    __properties = ["activationCode", "password", "name", "username", "privacyPolicy"]

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
    def from_json(cls, json_str: str) -> ActivateUserByThirdPartyActivationCodeRequest:
        """Create an instance of ActivateUserByThirdPartyActivationCodeRequest from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> ActivateUserByThirdPartyActivationCodeRequest:
        """Create an instance of ActivateUserByThirdPartyActivationCodeRequest from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return ActivateUserByThirdPartyActivationCodeRequest.construct(**obj)

        _obj = ActivateUserByThirdPartyActivationCodeRequest.construct(**{
            "activation_code": obj.get("activationCode"),
            "password": obj.get("password"),
            "name": obj.get("name"),
            "username": obj.get("username"),
            "privacy_policy": obj.get("privacyPolicy")
        })
        return _obj

