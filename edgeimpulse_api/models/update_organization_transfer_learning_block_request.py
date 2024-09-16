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


from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, StrictBool, StrictInt, StrictStr
from edgeimpulse_api.models.block_display_category import BlockDisplayCategory
from edgeimpulse_api.models.image_input_scaling import ImageInputScaling
from edgeimpulse_api.models.object_detection_last_layer import ObjectDetectionLastLayer
from edgeimpulse_api.models.organization_transfer_learning_block_custom_variant import OrganizationTransferLearningBlockCustomVariant
from edgeimpulse_api.models.organization_transfer_learning_operates_on import OrganizationTransferLearningOperatesOn
from edgeimpulse_api.models.public_project_tier_availability import PublicProjectTierAvailability

class UpdateOrganizationTransferLearningBlockRequest(BaseModel):
    name: Optional[StrictStr] = None
    docker_container: Optional[StrictStr] = Field(None, alias="dockerContainer")
    description: Optional[StrictStr] = None
    operates_on: Optional[OrganizationTransferLearningOperatesOn] = Field(None, alias="operatesOn")
    object_detection_last_layer: Optional[ObjectDetectionLastLayer] = Field(None, alias="objectDetectionLastLayer")
    implementation_version: Optional[StrictInt] = Field(None, alias="implementationVersion")
    is_public: Optional[StrictBool] = Field(None, alias="isPublic", description="Whether this block is publicly available to Edge Impulse users (if false, then only for members of the owning organization)")
    is_public_for_devices: Optional[List[StrictStr]] = Field(None, alias="isPublicForDevices", description="If `isPublic` is true, the list of devices (from latencyDevices) for which this model can be shown.")
    public_project_tier_availability: Optional[PublicProjectTierAvailability] = Field(None, alias="publicProjectTierAvailability")
    repository_url: Optional[StrictStr] = Field(None, alias="repositoryUrl", description="URL to the source code of this custom learn block.")
    parameters: Optional[List[Dict[str, Any]]] = Field(None, description="List of parameters, spec'ed according to https://docs.edgeimpulse.com/docs/tips-and-tricks/adding-parameters-to-custom-blocks")
    image_input_scaling: Optional[ImageInputScaling] = Field(None, alias="imageInputScaling")
    ind_requires_gpu: Optional[StrictBool] = Field(None, alias="indRequiresGpu", description="If set, requires this block to be scheduled on GPU.")
    display_category: Optional[BlockDisplayCategory] = Field(None, alias="displayCategory")
    custom_model_variants: Optional[List[OrganizationTransferLearningBlockCustomVariant]] = Field(None, alias="customModelVariants", description="List of custom model variants produced when this block is trained. This is experimental and may change in the future.")
    __properties = ["name", "dockerContainer", "description", "operatesOn", "objectDetectionLastLayer", "implementationVersion", "isPublic", "isPublicForDevices", "publicProjectTierAvailability", "repositoryUrl", "parameters", "imageInputScaling", "indRequiresGpu", "displayCategory", "customModelVariants"]

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
    def from_json(cls, json_str: str) -> UpdateOrganizationTransferLearningBlockRequest:
        """Create an instance of UpdateOrganizationTransferLearningBlockRequest from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of each item in custom_model_variants (list)
        _items = []
        if self.custom_model_variants:
            for _item in self.custom_model_variants:
                if _item:
                    _items.append(_item.to_dict())
            _dict['customModelVariants'] = _items
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> UpdateOrganizationTransferLearningBlockRequest:
        """Create an instance of UpdateOrganizationTransferLearningBlockRequest from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return UpdateOrganizationTransferLearningBlockRequest.construct(**obj)

        _obj = UpdateOrganizationTransferLearningBlockRequest.construct(**{
            "name": obj.get("name"),
            "docker_container": obj.get("dockerContainer"),
            "description": obj.get("description"),
            "operates_on": obj.get("operatesOn"),
            "object_detection_last_layer": obj.get("objectDetectionLastLayer"),
            "implementation_version": obj.get("implementationVersion"),
            "is_public": obj.get("isPublic"),
            "is_public_for_devices": obj.get("isPublicForDevices"),
            "public_project_tier_availability": obj.get("publicProjectTierAvailability"),
            "repository_url": obj.get("repositoryUrl"),
            "parameters": obj.get("parameters"),
            "image_input_scaling": obj.get("imageInputScaling"),
            "ind_requires_gpu": obj.get("indRequiresGpu"),
            "display_category": obj.get("displayCategory"),
            "custom_model_variants": [OrganizationTransferLearningBlockCustomVariant.from_dict(_item) for _item in obj.get("customModelVariants")] if obj.get("customModelVariants") is not None else None
        })
        return _obj

