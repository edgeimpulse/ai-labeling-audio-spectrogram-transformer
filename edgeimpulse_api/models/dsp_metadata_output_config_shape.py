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
from pydantic import BaseModel, Field, StrictInt

class DSPMetadataOutputConfigShape(BaseModel):
    width: StrictInt = Field(..., description="Available on all types. Denotes the width of an 'image' or 'spectrogram', or the number of features in a 'flat' block.")
    height: Optional[StrictInt] = Field(None, description="Only available for type 'image' and 'spectrogram'")
    channels: Optional[StrictInt] = Field(None, description="Only available for type 'image'")
    frames: Optional[StrictInt] = Field(None, description="Number of frames, only available for type 'image'")
    __properties = ["width", "height", "channels", "frames"]

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
    def from_json(cls, json_str: str) -> DSPMetadataOutputConfigShape:
        """Create an instance of DSPMetadataOutputConfigShape from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> DSPMetadataOutputConfigShape:
        """Create an instance of DSPMetadataOutputConfigShape from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return DSPMetadataOutputConfigShape.construct(**obj)

        _obj = DSPMetadataOutputConfigShape.construct(**{
            "width": obj.get("width"),
            "height": obj.get("height"),
            "channels": obj.get("channels"),
            "frames": obj.get("frames")
        })
        return _obj
