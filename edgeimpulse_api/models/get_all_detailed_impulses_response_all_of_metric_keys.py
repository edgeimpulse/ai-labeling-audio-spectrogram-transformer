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
from edgeimpulse_api.models.detailed_impulse_metric_filtering_type import DetailedImpulseMetricFilteringType

class GetAllDetailedImpulsesResponseAllOfMetricKeys(BaseModel):
    name: StrictStr = ...
    description: StrictStr = ...
    type: StrictStr = ...
    filtering_type: Optional[DetailedImpulseMetricFilteringType] = Field(None, alias="filteringType")
    show_in_table: StrictBool = Field(..., alias="showInTable")
    __properties = ["name", "description", "type", "filteringType", "showInTable"]

    @validator('type')
    def type_validate_enum(cls, v):
        if v not in ('core', 'additional'):
            raise ValueError("must validate the enum values ('core', 'additional')")
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
    def from_json(cls, json_str: str) -> GetAllDetailedImpulsesResponseAllOfMetricKeys:
        """Create an instance of GetAllDetailedImpulsesResponseAllOfMetricKeys from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of filtering_type
        if self.filtering_type:
            _dict['filteringType'] = self.filtering_type.to_dict()
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> GetAllDetailedImpulsesResponseAllOfMetricKeys:
        """Create an instance of GetAllDetailedImpulsesResponseAllOfMetricKeys from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return GetAllDetailedImpulsesResponseAllOfMetricKeys.construct(**obj)

        _obj = GetAllDetailedImpulsesResponseAllOfMetricKeys.construct(**{
            "name": obj.get("name"),
            "description": obj.get("description"),
            "type": obj.get("type"),
            "filtering_type": DetailedImpulseMetricFilteringType.from_dict(obj.get("filteringType")) if obj.get("filteringType") is not None else None,
            "show_in_table": obj.get("showInTable")
        })
        return _obj

