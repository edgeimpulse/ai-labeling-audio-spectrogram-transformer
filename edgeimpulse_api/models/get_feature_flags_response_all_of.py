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
from pydantic import BaseModel, Field
from edgeimpulse_api.models.get_feature_flags_response_all_of_flags import GetFeatureFlagsResponseAllOfFlags

class GetFeatureFlagsResponseAllOf(BaseModel):
    flags: List[GetFeatureFlagsResponseAllOfFlags] = Field(..., description="List of feature flags.")
    __properties = ["flags"]

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
    def from_json(cls, json_str: str) -> GetFeatureFlagsResponseAllOf:
        """Create an instance of GetFeatureFlagsResponseAllOf from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of each item in flags (list)
        _items = []
        if self.flags:
            for _item in self.flags:
                if _item:
                    _items.append(_item.to_dict())
            _dict['flags'] = _items
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> GetFeatureFlagsResponseAllOf:
        """Create an instance of GetFeatureFlagsResponseAllOf from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return GetFeatureFlagsResponseAllOf.construct(**obj)

        _obj = GetFeatureFlagsResponseAllOf.construct(**{
            "flags": [GetFeatureFlagsResponseAllOfFlags.from_dict(_item) for _item in obj.get("flags")] if obj.get("flags") is not None else None
        })
        return _obj
