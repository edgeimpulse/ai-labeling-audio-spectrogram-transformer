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
from pydantic import BaseModel, Field, StrictInt, StrictStr

class UpdateOrganizationDspBlockRequest(BaseModel):
    name: Optional[StrictStr] = None
    docker_container: Optional[StrictStr] = Field(None, alias="dockerContainer")
    description: Optional[StrictStr] = None
    requests_cpu: Optional[float] = Field(None, alias="requestsCpu")
    requests_memory: Optional[StrictInt] = Field(None, alias="requestsMemory")
    limits_cpu: Optional[float] = Field(None, alias="limitsCpu")
    limits_memory: Optional[StrictInt] = Field(None, alias="limitsMemory")
    port: Optional[StrictInt] = None
    __properties = ["name", "dockerContainer", "description", "requestsCpu", "requestsMemory", "limitsCpu", "limitsMemory", "port"]

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
    def from_json(cls, json_str: str) -> UpdateOrganizationDspBlockRequest:
        """Create an instance of UpdateOrganizationDspBlockRequest from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> UpdateOrganizationDspBlockRequest:
        """Create an instance of UpdateOrganizationDspBlockRequest from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return UpdateOrganizationDspBlockRequest.construct(**obj)

        _obj = UpdateOrganizationDspBlockRequest.construct(**{
            "name": obj.get("name"),
            "docker_container": obj.get("dockerContainer"),
            "description": obj.get("description"),
            "requests_cpu": obj.get("requestsCpu"),
            "requests_memory": obj.get("requestsMemory"),
            "limits_cpu": obj.get("limitsCpu"),
            "limits_memory": obj.get("limitsMemory"),
            "port": obj.get("port")
        })
        return _obj

