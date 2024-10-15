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



from pydantic import BaseModel, Field, StrictInt, StrictStr

class AutoLabelerSegment(BaseModel):
    id: StrictInt = ...
    mask_url: StrictStr = Field(..., alias="maskUrl")
    mask_x: StrictInt = Field(..., alias="maskX")
    mask_y: StrictInt = Field(..., alias="maskY")
    mask_width: StrictInt = Field(..., alias="maskWidth")
    mask_height: StrictInt = Field(..., alias="maskHeight")
    cluster: StrictInt = ...
    __properties = ["id", "maskUrl", "maskX", "maskY", "maskWidth", "maskHeight", "cluster"]

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
    def from_json(cls, json_str: str) -> AutoLabelerSegment:
        """Create an instance of AutoLabelerSegment from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> AutoLabelerSegment:
        """Create an instance of AutoLabelerSegment from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return AutoLabelerSegment.construct(**obj)

        _obj = AutoLabelerSegment.construct(**{
            "id": obj.get("id"),
            "mask_url": obj.get("maskUrl"),
            "mask_x": obj.get("maskX"),
            "mask_y": obj.get("maskY"),
            "mask_width": obj.get("maskWidth"),
            "mask_height": obj.get("maskHeight"),
            "cluster": obj.get("cluster")
        })
        return _obj
