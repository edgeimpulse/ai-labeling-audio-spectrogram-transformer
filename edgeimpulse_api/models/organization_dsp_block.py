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

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, StrictBool, StrictInt, StrictStr
from edgeimpulse_api.models.created_updated_by_user import CreatedUpdatedByUser

class OrganizationDspBlock(BaseModel):
    id: StrictInt = ...
    name: StrictStr = ...
    docker_container: StrictStr = Field(..., alias="dockerContainer")
    docker_container_managed_by_edge_impulse: StrictBool = Field(..., alias="dockerContainerManagedByEdgeImpulse")
    created: datetime = ...
    created_by_user: Optional[CreatedUpdatedByUser] = Field(None, alias="createdByUser")
    last_updated: Optional[datetime] = Field(None, alias="lastUpdated")
    last_updated_by_user: Optional[CreatedUpdatedByUser] = Field(None, alias="lastUpdatedByUser")
    user_id: Optional[StrictInt] = Field(None, alias="userId")
    user_name: Optional[StrictStr] = Field(None, alias="userName")
    description: StrictStr = ...
    requests_cpu: Optional[float] = Field(None, alias="requestsCpu")
    requests_memory: Optional[StrictInt] = Field(None, alias="requestsMemory")
    limits_cpu: Optional[float] = Field(None, alias="limitsCpu")
    limits_memory: Optional[StrictInt] = Field(None, alias="limitsMemory")
    port: StrictInt = ...
    is_connected: StrictBool = Field(..., alias="isConnected")
    error: Optional[StrictStr] = None
    source_code_available: StrictBool = Field(..., alias="sourceCodeAvailable")
    __properties = ["id", "name", "dockerContainer", "dockerContainerManagedByEdgeImpulse", "created", "createdByUser", "lastUpdated", "lastUpdatedByUser", "userId", "userName", "description", "requestsCpu", "requestsMemory", "limitsCpu", "limitsMemory", "port", "isConnected", "error", "sourceCodeAvailable"]

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
    def from_json(cls, json_str: str) -> OrganizationDspBlock:
        """Create an instance of OrganizationDspBlock from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of created_by_user
        if self.created_by_user:
            _dict['createdByUser'] = self.created_by_user.to_dict()
        # override the default output from pydantic by calling `to_dict()` of last_updated_by_user
        if self.last_updated_by_user:
            _dict['lastUpdatedByUser'] = self.last_updated_by_user.to_dict()
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> OrganizationDspBlock:
        """Create an instance of OrganizationDspBlock from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return OrganizationDspBlock.construct(**obj)

        _obj = OrganizationDspBlock.construct(**{
            "id": obj.get("id"),
            "name": obj.get("name"),
            "docker_container": obj.get("dockerContainer"),
            "docker_container_managed_by_edge_impulse": obj.get("dockerContainerManagedByEdgeImpulse"),
            "created": obj.get("created"),
            "created_by_user": CreatedUpdatedByUser.from_dict(obj.get("createdByUser")) if obj.get("createdByUser") is not None else None,
            "last_updated": obj.get("lastUpdated"),
            "last_updated_by_user": CreatedUpdatedByUser.from_dict(obj.get("lastUpdatedByUser")) if obj.get("lastUpdatedByUser") is not None else None,
            "user_id": obj.get("userId"),
            "user_name": obj.get("userName"),
            "description": obj.get("description"),
            "requests_cpu": obj.get("requestsCpu"),
            "requests_memory": obj.get("requestsMemory"),
            "limits_cpu": obj.get("limitsCpu"),
            "limits_memory": obj.get("limitsMemory"),
            "port": obj.get("port"),
            "is_connected": obj.get("isConnected"),
            "error": obj.get("error"),
            "source_code_available": obj.get("sourceCodeAvailable")
        })
        return _obj
