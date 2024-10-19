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
from typing import List, Optional
from pydantic import BaseModel, Field, StrictBool, StrictStr, validator
from edgeimpulse_api.models.image_input_scaling import ImageInputScaling
from edgeimpulse_api.models.keras_model_layer import KerasModelLayer
from edgeimpulse_api.models.keras_model_metadata_metrics import KerasModelMetadataMetrics
from edgeimpulse_api.models.keras_model_type_enum import KerasModelTypeEnum
from edgeimpulse_api.models.object_detection_last_layer import ObjectDetectionLastLayer

class KerasModelMetadata(BaseModel):
    created: datetime = Field(..., description="Date when the model was trained")
    layers: List[KerasModelLayer] = Field(..., description="Layers of the neural network")
    class_names: List[StrictStr] = Field(..., alias="classNames", description="Labels for the output layer")
    labels: List[StrictStr] = Field(..., description="Original labels in the dataset when features were generated, e.g. used to render the feature explorer.")
    available_model_types: List[KerasModelTypeEnum] = Field(..., alias="availableModelTypes", description="The types of model that are available")
    recommended_model_type: KerasModelTypeEnum = Field(..., alias="recommendedModelType")
    model_validation_metrics: List[KerasModelMetadataMetrics] = Field(..., alias="modelValidationMetrics", description="Metrics for each of the available model types")
    has_trained_model: StrictBool = Field(..., alias="hasTrainedModel")
    mode: StrictStr = ...
    object_detection_last_layer: Optional[ObjectDetectionLastLayer] = Field(None, alias="objectDetectionLastLayer")
    image_input_scaling: ImageInputScaling = Field(..., alias="imageInputScaling")
    __properties = ["created", "layers", "classNames", "labels", "availableModelTypes", "recommendedModelType", "modelValidationMetrics", "hasTrainedModel", "mode", "objectDetectionLastLayer", "imageInputScaling"]

    @validator('mode')
    def mode_validate_enum(cls, v):
        if v not in ('classification', 'regression', 'object-detection', 'visual-anomaly', 'anomaly-gmm'):
            raise ValueError("must validate the enum values ('classification', 'regression', 'object-detection', 'visual-anomaly', 'anomaly-gmm')")
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
    def from_json(cls, json_str: str) -> KerasModelMetadata:
        """Create an instance of KerasModelMetadata from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of each item in layers (list)
        _items = []
        if self.layers:
            for _item in self.layers:
                if _item:
                    _items.append(_item.to_dict())
            _dict['layers'] = _items
        # override the default output from pydantic by calling `to_dict()` of each item in model_validation_metrics (list)
        _items = []
        if self.model_validation_metrics:
            for _item in self.model_validation_metrics:
                if _item:
                    _items.append(_item.to_dict())
            _dict['modelValidationMetrics'] = _items
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> KerasModelMetadata:
        """Create an instance of KerasModelMetadata from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return KerasModelMetadata.construct(**obj)

        _obj = KerasModelMetadata.construct(**{
            "created": obj.get("created"),
            "layers": [KerasModelLayer.from_dict(_item) for _item in obj.get("layers")] if obj.get("layers") is not None else None,
            "class_names": obj.get("classNames"),
            "labels": obj.get("labels"),
            "available_model_types": obj.get("availableModelTypes"),
            "recommended_model_type": obj.get("recommendedModelType"),
            "model_validation_metrics": [KerasModelMetadataMetrics.from_dict(_item) for _item in obj.get("modelValidationMetrics")] if obj.get("modelValidationMetrics") is not None else None,
            "has_trained_model": obj.get("hasTrainedModel"),
            "mode": obj.get("mode"),
            "object_detection_last_layer": obj.get("objectDetectionLastLayer"),
            "image_input_scaling": obj.get("imageInputScaling")
        })
        return _obj

