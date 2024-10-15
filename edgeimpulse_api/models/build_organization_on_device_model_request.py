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
from edgeimpulse_api.models.deployment_target_engine import DeploymentTargetEngine
from edgeimpulse_api.models.keras_model_type_enum import KerasModelTypeEnum

class BuildOrganizationOnDeviceModelRequest(BaseModel):
    engine: DeploymentTargetEngine = ...
    deploy_block_id: StrictInt = Field(..., alias="deployBlockId")
    model_type: Optional[KerasModelTypeEnum] = Field(None, alias="modelType")
    __properties = ["engine", "deployBlockId", "modelType"]

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
    def from_json(cls, json_str: str) -> BuildOrganizationOnDeviceModelRequest:
        """Create an instance of BuildOrganizationOnDeviceModelRequest from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> BuildOrganizationOnDeviceModelRequest:
        """Create an instance of BuildOrganizationOnDeviceModelRequest from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return BuildOrganizationOnDeviceModelRequest.construct(**obj)

        _obj = BuildOrganizationOnDeviceModelRequest.construct(**{
            "engine": obj.get("engine"),
            "deploy_block_id": obj.get("deployBlockId"),
            "model_type": obj.get("modelType")
        })
        return _obj
