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


from typing import List, Optional
from pydantic import BaseModel, Field, StrictBool, StrictInt, StrictStr, validator
from edgeimpulse_api.models.deployment_target_badge import DeploymentTargetBadge
from edgeimpulse_api.models.deployment_target_engine import DeploymentTargetEngine

class DeploymentTarget(BaseModel):
    name: StrictStr = ...
    description: StrictStr = ...
    image: StrictStr = ...
    image_classes: StrictStr = Field(..., alias="imageClasses")
    format: StrictStr = ...
    latency_device: Optional[StrictStr] = Field(None, alias="latencyDevice")
    has_eon_compiler: StrictBool = Field(..., alias="hasEonCompiler", description="Preferably use supportedEngines / preferredEngine")
    has_tensor_rt: StrictBool = Field(..., alias="hasTensorRT", description="Preferably use supportedEngines / preferredEngine")
    has_tensai_flow: StrictBool = Field(..., alias="hasTensaiFlow", description="Preferably use supportedEngines / preferredEngine")
    has_drpai: StrictBool = Field(..., alias="hasDRPAI", description="Preferably use supportedEngines / preferredEngine")
    has_tidl: StrictBool = Field(..., alias="hasTIDL", description="Preferably use supportedEngines / preferredEngine")
    has_akida: StrictBool = Field(..., alias="hasAkida", description="Preferably use supportedEngines / preferredEngine")
    has_memryx: StrictBool = Field(..., alias="hasMemryx", description="Preferably use supportedEngines / preferredEngine")
    hide_optimizations: StrictBool = Field(..., alias="hideOptimizations")
    badge: Optional[DeploymentTargetBadge] = None
    ui_section: StrictStr = Field(..., alias="uiSection")
    custom_deploy_id: Optional[StrictInt] = Field(None, alias="customDeployId")
    integrate_url: Optional[StrictStr] = Field(None, alias="integrateUrl")
    owner_organization_name: Optional[StrictStr] = Field(None, alias="ownerOrganizationName")
    supported_engines: List[DeploymentTargetEngine] = Field(..., alias="supportedEngines")
    preferred_engine: DeploymentTargetEngine = Field(..., alias="preferredEngine")
    url: Optional[StrictStr] = None
    docs_url: StrictStr = Field(..., alias="docsUrl")
    firmware_repo_url: Optional[StrictStr] = Field(None, alias="firmwareRepoUrl")
    __properties = ["name", "description", "image", "imageClasses", "format", "latencyDevice", "hasEonCompiler", "hasTensorRT", "hasTensaiFlow", "hasDRPAI", "hasTIDL", "hasAkida", "hasMemryx", "hideOptimizations", "badge", "uiSection", "customDeployId", "integrateUrl", "ownerOrganizationName", "supportedEngines", "preferredEngine", "url", "docsUrl", "firmwareRepoUrl"]

    @validator('ui_section')
    def ui_section_validate_enum(cls, v):
        if v not in ('library', 'firmware', 'mobile', 'hidden'):
            raise ValueError("must validate the enum values ('library', 'firmware', 'mobile', 'hidden')")
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
    def from_json(cls, json_str: str) -> DeploymentTarget:
        """Create an instance of DeploymentTarget from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of badge
        if self.badge:
            _dict['badge'] = self.badge.to_dict()
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> DeploymentTarget:
        """Create an instance of DeploymentTarget from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return DeploymentTarget.construct(**obj)

        _obj = DeploymentTarget.construct(**{
            "name": obj.get("name"),
            "description": obj.get("description"),
            "image": obj.get("image"),
            "image_classes": obj.get("imageClasses"),
            "format": obj.get("format"),
            "latency_device": obj.get("latencyDevice"),
            "has_eon_compiler": obj.get("hasEonCompiler"),
            "has_tensor_rt": obj.get("hasTensorRT"),
            "has_tensai_flow": obj.get("hasTensaiFlow"),
            "has_drpai": obj.get("hasDRPAI"),
            "has_tidl": obj.get("hasTIDL"),
            "has_akida": obj.get("hasAkida"),
            "has_memryx": obj.get("hasMemryx"),
            "hide_optimizations": obj.get("hideOptimizations"),
            "badge": DeploymentTargetBadge.from_dict(obj.get("badge")) if obj.get("badge") is not None else None,
            "ui_section": obj.get("uiSection"),
            "custom_deploy_id": obj.get("customDeployId"),
            "integrate_url": obj.get("integrateUrl"),
            "owner_organization_name": obj.get("ownerOrganizationName"),
            "supported_engines": obj.get("supportedEngines"),
            "preferred_engine": obj.get("preferredEngine"),
            "url": obj.get("url"),
            "docs_url": obj.get("docsUrl"),
            "firmware_repo_url": obj.get("firmwareRepoUrl")
        })
        return _obj

