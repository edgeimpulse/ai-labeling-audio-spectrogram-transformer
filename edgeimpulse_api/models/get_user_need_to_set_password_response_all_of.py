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


from typing import List, Optional
from pydantic import BaseModel, Field, StrictBool, StrictStr
from edgeimpulse_api.models.enterprise_trial import EnterpriseTrial

class GetUserNeedToSetPasswordResponseAllOf(BaseModel):
    email: Optional[StrictStr] = Field(None, description="User email")
    need_password: Optional[StrictBool] = Field(None, alias="needPassword", description="Whether the user needs to set its password or not")
    whitelabels: Optional[List[StrictStr]] = Field(None, description="White label domains the user belongs to, if any")
    trials: Optional[List[EnterpriseTrial]] = Field(None, description="Current or past enterprise trials.")
    email_verified: Optional[StrictBool] = Field(None, alias="emailVerified", description="Whether the user has verified its email address or not")
    __properties = ["email", "needPassword", "whitelabels", "trials", "emailVerified"]

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
    def from_json(cls, json_str: str) -> GetUserNeedToSetPasswordResponseAllOf:
        """Create an instance of GetUserNeedToSetPasswordResponseAllOf from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of each item in trials (list)
        _items = []
        if self.trials:
            for _item in self.trials:
                if _item:
                    _items.append(_item.to_dict())
            _dict['trials'] = _items
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> GetUserNeedToSetPasswordResponseAllOf:
        """Create an instance of GetUserNeedToSetPasswordResponseAllOf from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return GetUserNeedToSetPasswordResponseAllOf.construct(**obj)

        _obj = GetUserNeedToSetPasswordResponseAllOf.construct(**{
            "email": obj.get("email"),
            "need_password": obj.get("needPassword"),
            "whitelabels": obj.get("whitelabels"),
            "trials": [EnterpriseTrial.from_dict(_item) for _item in obj.get("trials")] if obj.get("trials") is not None else None,
            "email_verified": obj.get("emailVerified")
        })
        return _obj

