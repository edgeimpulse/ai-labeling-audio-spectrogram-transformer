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
from pydantic import BaseModel, Field
from edgeimpulse_api.models.resource_range import ResourceRange
from edgeimpulse_api.models.target_memory import TargetMemory

class ApplicationBudget(BaseModel):
    latency_per_inference_ms: Optional[ResourceRange] = Field(None, alias="latencyPerInferenceMs")
    energy_per_inference_joules: Optional[ResourceRange] = Field(None, alias="energyPerInferenceJoules")
    memory_overhead: Optional[TargetMemory] = Field(None, alias="memoryOverhead")
    __properties = ["latencyPerInferenceMs", "energyPerInferenceJoules", "memoryOverhead"]

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
    def from_json(cls, json_str: str) -> ApplicationBudget:
        """Create an instance of ApplicationBudget from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of latency_per_inference_ms
        if self.latency_per_inference_ms:
            _dict['latencyPerInferenceMs'] = self.latency_per_inference_ms.to_dict()
        # override the default output from pydantic by calling `to_dict()` of energy_per_inference_joules
        if self.energy_per_inference_joules:
            _dict['energyPerInferenceJoules'] = self.energy_per_inference_joules.to_dict()
        # override the default output from pydantic by calling `to_dict()` of memory_overhead
        if self.memory_overhead:
            _dict['memoryOverhead'] = self.memory_overhead.to_dict()
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> ApplicationBudget:
        """Create an instance of ApplicationBudget from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return ApplicationBudget.construct(**obj)

        _obj = ApplicationBudget.construct(**{
            "latency_per_inference_ms": ResourceRange.from_dict(obj.get("latencyPerInferenceMs")) if obj.get("latencyPerInferenceMs") is not None else None,
            "energy_per_inference_joules": ResourceRange.from_dict(obj.get("energyPerInferenceJoules")) if obj.get("energyPerInferenceJoules") is not None else None,
            "memory_overhead": TargetMemory.from_dict(obj.get("memoryOverhead")) if obj.get("memoryOverhead") is not None else None
        })
        return _obj
