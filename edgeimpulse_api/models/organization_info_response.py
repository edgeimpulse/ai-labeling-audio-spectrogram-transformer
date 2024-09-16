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
from pydantic import BaseModel, Field, StrictBool, StrictInt, StrictStr
from edgeimpulse_api.models.entitlement_limits import EntitlementLimits
from edgeimpulse_api.models.organization import Organization
from edgeimpulse_api.models.organization_dataset import OrganizationDataset
from edgeimpulse_api.models.organization_info_response_all_of_cli_lists import OrganizationInfoResponseAllOfCliLists
from edgeimpulse_api.models.organization_info_response_all_of_default_compute_limits import OrganizationInfoResponseAllOfDefaultComputeLimits
from edgeimpulse_api.models.organization_info_response_all_of_performance import OrganizationInfoResponseAllOfPerformance
from edgeimpulse_api.models.project_info_response_all_of_experiments import ProjectInfoResponseAllOfExperiments
from edgeimpulse_api.models.project_info_response_all_of_readme import ProjectInfoResponseAllOfReadme

class OrganizationInfoResponse(BaseModel):
    success: StrictBool = Field(..., description="Whether the operation succeeded")
    error: Optional[StrictStr] = Field(None, description="Optional error description (set if 'success' was false)")
    organization: Organization = ...
    datasets: List[OrganizationDataset] = ...
    default_compute_limits: OrganizationInfoResponseAllOfDefaultComputeLimits = Field(..., alias="defaultComputeLimits")
    entitlement_limits: Optional[EntitlementLimits] = Field(None, alias="entitlementLimits")
    experiments: List[ProjectInfoResponseAllOfExperiments] = Field(..., description="Experiments that the organization has access to. Enabling experiments can only be done through a JWT token.")
    readme: Optional[ProjectInfoResponseAllOfReadme] = None
    whitelabel_id: Optional[StrictInt] = Field(None, alias="whitelabelId")
    cli_lists: OrganizationInfoResponseAllOfCliLists = Field(..., alias="cliLists")
    performance: OrganizationInfoResponseAllOfPerformance = ...
    __properties = ["success", "error", "organization", "datasets", "defaultComputeLimits", "entitlementLimits", "experiments", "readme", "whitelabelId", "cliLists", "performance"]

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
    def from_json(cls, json_str: str) -> OrganizationInfoResponse:
        """Create an instance of OrganizationInfoResponse from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of organization
        if self.organization:
            _dict['organization'] = self.organization.to_dict()
        # override the default output from pydantic by calling `to_dict()` of each item in datasets (list)
        _items = []
        if self.datasets:
            for _item in self.datasets:
                if _item:
                    _items.append(_item.to_dict())
            _dict['datasets'] = _items
        # override the default output from pydantic by calling `to_dict()` of default_compute_limits
        if self.default_compute_limits:
            _dict['defaultComputeLimits'] = self.default_compute_limits.to_dict()
        # override the default output from pydantic by calling `to_dict()` of entitlement_limits
        if self.entitlement_limits:
            _dict['entitlementLimits'] = self.entitlement_limits.to_dict()
        # override the default output from pydantic by calling `to_dict()` of each item in experiments (list)
        _items = []
        if self.experiments:
            for _item in self.experiments:
                if _item:
                    _items.append(_item.to_dict())
            _dict['experiments'] = _items
        # override the default output from pydantic by calling `to_dict()` of readme
        if self.readme:
            _dict['readme'] = self.readme.to_dict()
        # override the default output from pydantic by calling `to_dict()` of cli_lists
        if self.cli_lists:
            _dict['cliLists'] = self.cli_lists.to_dict()
        # override the default output from pydantic by calling `to_dict()` of performance
        if self.performance:
            _dict['performance'] = self.performance.to_dict()
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> OrganizationInfoResponse:
        """Create an instance of OrganizationInfoResponse from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return OrganizationInfoResponse.construct(**obj)

        _obj = OrganizationInfoResponse.construct(**{
            "success": obj.get("success"),
            "error": obj.get("error"),
            "organization": Organization.from_dict(obj.get("organization")) if obj.get("organization") is not None else None,
            "datasets": [OrganizationDataset.from_dict(_item) for _item in obj.get("datasets")] if obj.get("datasets") is not None else None,
            "default_compute_limits": OrganizationInfoResponseAllOfDefaultComputeLimits.from_dict(obj.get("defaultComputeLimits")) if obj.get("defaultComputeLimits") is not None else None,
            "entitlement_limits": EntitlementLimits.from_dict(obj.get("entitlementLimits")) if obj.get("entitlementLimits") is not None else None,
            "experiments": [ProjectInfoResponseAllOfExperiments.from_dict(_item) for _item in obj.get("experiments")] if obj.get("experiments") is not None else None,
            "readme": ProjectInfoResponseAllOfReadme.from_dict(obj.get("readme")) if obj.get("readme") is not None else None,
            "whitelabel_id": obj.get("whitelabelId"),
            "cli_lists": OrganizationInfoResponseAllOfCliLists.from_dict(obj.get("cliLists")) if obj.get("cliLists") is not None else None,
            "performance": OrganizationInfoResponseAllOfPerformance.from_dict(obj.get("performance")) if obj.get("performance") is not None else None
        })
        return _obj

