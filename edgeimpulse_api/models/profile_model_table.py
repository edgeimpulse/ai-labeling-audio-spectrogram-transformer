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
from edgeimpulse_api.models.profile_model_table_mcu import ProfileModelTableMcu
from edgeimpulse_api.models.profile_model_table_mpu import ProfileModelTableMpu

class ProfileModelTable(BaseModel):
    variant: StrictStr = ...
    low_end_mcu: ProfileModelTableMcu = Field(..., alias="lowEndMcu")
    high_end_mcu: ProfileModelTableMcu = Field(..., alias="highEndMcu")
    high_end_mcu_plus_accelerator: ProfileModelTableMcu = Field(..., alias="highEndMcuPlusAccelerator")
    mpu: ProfileModelTableMpu = ...
    gpu_or_mpu_accelerator: ProfileModelTableMpu = Field(..., alias="gpuOrMpuAccelerator")
    __properties = ["variant", "lowEndMcu", "highEndMcu", "highEndMcuPlusAccelerator", "mpu", "gpuOrMpuAccelerator"]

    @validator('variant')
    def variant_validate_enum(cls, v):
        if v not in ('int8', 'float32'):
            raise ValueError("must validate the enum values ('int8', 'float32')")
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
    def from_json(cls, json_str: str) -> ProfileModelTable:
        """Create an instance of ProfileModelTable from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of low_end_mcu
        if self.low_end_mcu:
            _dict['lowEndMcu'] = self.low_end_mcu.to_dict()
        # override the default output from pydantic by calling `to_dict()` of high_end_mcu
        if self.high_end_mcu:
            _dict['highEndMcu'] = self.high_end_mcu.to_dict()
        # override the default output from pydantic by calling `to_dict()` of high_end_mcu_plus_accelerator
        if self.high_end_mcu_plus_accelerator:
            _dict['highEndMcuPlusAccelerator'] = self.high_end_mcu_plus_accelerator.to_dict()
        # override the default output from pydantic by calling `to_dict()` of mpu
        if self.mpu:
            _dict['mpu'] = self.mpu.to_dict()
        # override the default output from pydantic by calling `to_dict()` of gpu_or_mpu_accelerator
        if self.gpu_or_mpu_accelerator:
            _dict['gpuOrMpuAccelerator'] = self.gpu_or_mpu_accelerator.to_dict()
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> ProfileModelTable:
        """Create an instance of ProfileModelTable from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return ProfileModelTable.construct(**obj)

        _obj = ProfileModelTable.construct(**{
            "variant": obj.get("variant"),
            "low_end_mcu": ProfileModelTableMcu.from_dict(obj.get("lowEndMcu")) if obj.get("lowEndMcu") is not None else None,
            "high_end_mcu": ProfileModelTableMcu.from_dict(obj.get("highEndMcu")) if obj.get("highEndMcu") is not None else None,
            "high_end_mcu_plus_accelerator": ProfileModelTableMcu.from_dict(obj.get("highEndMcuPlusAccelerator")) if obj.get("highEndMcuPlusAccelerator") is not None else None,
            "mpu": ProfileModelTableMpu.from_dict(obj.get("mpu")) if obj.get("mpu") is not None else None,
            "gpu_or_mpu_accelerator": ProfileModelTableMpu.from_dict(obj.get("gpuOrMpuAccelerator")) if obj.get("gpuOrMpuAccelerator") is not None else None
        })
        return _obj
