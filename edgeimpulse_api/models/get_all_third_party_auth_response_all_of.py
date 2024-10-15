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
from pydantic import BaseModel
from edgeimpulse_api.models.third_party_auth import ThirdPartyAuth

class GetAllThirdPartyAuthResponseAllOf(BaseModel):
    auths: List[ThirdPartyAuth] = ...
    __properties = ["auths"]

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
    def from_json(cls, json_str: str) -> GetAllThirdPartyAuthResponseAllOf:
        """Create an instance of GetAllThirdPartyAuthResponseAllOf from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of each item in auths (list)
        _items = []
        if self.auths:
            for _item in self.auths:
                if _item:
                    _items.append(_item.to_dict())
            _dict['auths'] = _items
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> GetAllThirdPartyAuthResponseAllOf:
        """Create an instance of GetAllThirdPartyAuthResponseAllOf from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return GetAllThirdPartyAuthResponseAllOf.construct(**obj)

        _obj = GetAllThirdPartyAuthResponseAllOf.construct(**{
            "auths": [ThirdPartyAuth.from_dict(_item) for _item in obj.get("auths")] if obj.get("auths") is not None else None
        })
        return _obj
