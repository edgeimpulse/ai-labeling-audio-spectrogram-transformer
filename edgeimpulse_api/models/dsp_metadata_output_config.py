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



from pydantic import BaseModel, Field, StrictStr, validator
from edgeimpulse_api.models.dsp_metadata_output_config_shape import DSPMetadataOutputConfigShape

class DSPMetadataOutputConfig(BaseModel):
    type: StrictStr = Field(..., description="Output type of the DSP block")
    shape: DSPMetadataOutputConfigShape = ...
    __properties = ["type", "shape"]

    @validator('type')
    def type_validate_enum(cls, v):
        if v not in ('image', 'spectrogram', 'flat'):
            raise ValueError("must validate the enum values ('image', 'spectrogram', 'flat')")
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
    def from_json(cls, json_str: str) -> DSPMetadataOutputConfig:
        """Create an instance of DSPMetadataOutputConfig from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of shape
        if self.shape:
            _dict['shape'] = self.shape.to_dict()
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> DSPMetadataOutputConfig:
        """Create an instance of DSPMetadataOutputConfig from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return DSPMetadataOutputConfig.construct(**obj)

        _obj = DSPMetadataOutputConfig.construct(**{
            "type": obj.get("type"),
            "shape": DSPMetadataOutputConfigShape.from_dict(obj.get("shape")) if obj.get("shape") is not None else None
        })
        return _obj

