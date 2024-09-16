# coding: utf-8

"""
    Edge Impulse API

    No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)  # noqa: E501

    The version of the OpenAPI document: 1.0.0
    Generated by: https://openapi-generator.tech
"""


from __future__ import absolute_import

import re  # noqa: F401

from pydantic import validate_arguments, ValidationError
from typing_extensions import Annotated

from pydantic import Field, StrictInt

from edgeimpulse_api.models.generic_api_response import GenericApiResponse
from edgeimpulse_api.models.get_impulse_blocks_response import GetImpulseBlocksResponse
from edgeimpulse_api.models.get_whitelabel_domain_response import GetWhitelabelDomainResponse
from edgeimpulse_api.models.update_whitelabel_deployment_targets_request import UpdateWhitelabelDeploymentTargetsRequest

from edgeimpulse_api.api_client import ApiClient
from edgeimpulse_api.exceptions import (  # noqa: F401
    ApiTypeError,
    ApiValueError
)


class WhitelabelsApi(object):
    def __init__(self, api_client=None):
        if api_client is None:
            api_client = ApiClient.get_default()
        self.api_client = api_client

    @validate_arguments
    def get_all_impulse_blocks(self, whitelabel_identifier : Annotated[StrictInt, Field(..., description="Whitelabel ID")], **kwargs) -> GetImpulseBlocksResponse:  # noqa: E501
        """Get impulse blocks

        Lists all possible DSP and ML blocks available for this white label.

        :param whitelabel_identifier: Whitelabel ID (required)
        :type whitelabel_identifier: int
        :param async_req: Whether to execute the request asynchronously.
        :type async_req: bool, optional
        :param _preload_content: if False, the urllib3.HTTPResponse object will
                                 be returned without reading/decoding response
                                 data. Default is True.
        :type _preload_content: bool, optional
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :return: Returns the result object.
                 If the method is called asynchronously,
                 returns the request thread.
        :rtype: GetImpulseBlocksResponse
        """
        kwargs['_return_http_data_only'] = True
        return self._get_all_impulse_blocks_with_http_info(whitelabel_identifier, **kwargs)  # noqa: E501

    @validate_arguments
    def _get_all_impulse_blocks_with_http_info(self, whitelabel_identifier : Annotated[StrictInt, Field(..., description="Whitelabel ID")], **kwargs):  # noqa: E501
        """Get impulse blocks 

        Lists all possible DSP and ML blocks available for this white label.

        :param whitelabel_identifier: Whitelabel ID (required)
        :type whitelabel_identifier: int
        :param async_req: Whether to execute the request asynchronously.
        :type async_req: bool, optional
        :param _return_http_data_only: response data without head status code
                                       and headers
        :type _return_http_data_only: bool, optional
        :param _preload_content: if False, the urllib3.HTTPResponse object will
                                 be returned without reading/decoding response
                                 data. Default is True.
        :type _preload_content: bool, optional
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :param _request_auth: set to override the auth_settings for an a single
                              request; this effectively ignores the authentication
                              in the spec for a single request.
        :type _request_auth: dict, optional
        :type _content_type: string, optional: force content-type for the request
        :return: Returns the result object.
                 If the method is called asynchronously,
                 returns the request thread.
        :rtype: tuple(GetImpulseBlocksResponse, status_code(int), headers(HTTPHeaderDict))
        """

        _params = locals()

        _all_params = [
            'whitelabel_identifier'
        ]
        _all_params.extend(
            [
                'async_req',
                '_return_http_data_only',
                '_preload_content',
                '_request_timeout',
                '_request_auth',
                '_content_type',
                '_headers'
            ]
        )

        # validate the arguments
        for _key, _val in _params['kwargs'].items():
            if _key not in _all_params:
                raise ApiTypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method get_all_impulse_blocks" % _key
                )
            _params[_key] = _val
        del _params['kwargs']

        _collection_formats = {}

        # process the path parameters
        _path_params = {}
        if _params['whitelabel_identifier']:
            _path_params['whitelabelIdentifier'] = _params['whitelabel_identifier']

        # process the query parameters
        _query_params = []

        # process the header parameters
        _header_params = dict(_params.get('_headers', {}))

        # process the form parameters
        _form_params = []
        _files = {}

        # process the body parameter
        _body_params = None

        # set the HTTP header `Accept`
        _header_params['Accept'] = self.api_client.select_header_accept(
            ['application/json'])  # noqa: E501

        # authentication setting
        _auth_settings = ['ApiKeyAuthentication', 'JWTAuthentication', 'JWTHttpHeaderAuthentication']  # noqa: E501

        _response_types_map = {
            '200': "GetImpulseBlocksResponse",
        }

        return self.api_client.call_api(
            '/api/whitelabel/{whitelabelIdentifier}/impulse/blocks', 'GET',
            _path_params,
            _query_params,
            _header_params,
            body=_body_params,
            post_params=_form_params,
            files=_files,
            response_types_map=_response_types_map,
            auth_settings=_auth_settings,
            async_req=_params.get('async_req'),
            _return_http_data_only=_params.get('_return_http_data_only'),  # noqa: E501
            _preload_content=_params.get('_preload_content', True),
            _request_timeout=_params.get('_request_timeout'),
            collection_formats=_collection_formats,
            _request_auth=_params.get('_request_auth'))

    @validate_arguments
    def get_whitelabel_domain(self, whitelabel_identifier : Annotated[StrictInt, Field(..., description="Whitelabel ID")], **kwargs) -> GetWhitelabelDomainResponse:  # noqa: E501
        """Get white label domain

        Get a white label domain given its unique identifier.

        :param whitelabel_identifier: Whitelabel ID (required)
        :type whitelabel_identifier: int
        :param async_req: Whether to execute the request asynchronously.
        :type async_req: bool, optional
        :param _preload_content: if False, the urllib3.HTTPResponse object will
                                 be returned without reading/decoding response
                                 data. Default is True.
        :type _preload_content: bool, optional
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :return: Returns the result object.
                 If the method is called asynchronously,
                 returns the request thread.
        :rtype: GetWhitelabelDomainResponse
        """
        kwargs['_return_http_data_only'] = True
        return self._get_whitelabel_domain_with_http_info(whitelabel_identifier, **kwargs)  # noqa: E501

    @validate_arguments
    def _get_whitelabel_domain_with_http_info(self, whitelabel_identifier : Annotated[StrictInt, Field(..., description="Whitelabel ID")], **kwargs):  # noqa: E501
        """Get white label domain 

        Get a white label domain given its unique identifier.

        :param whitelabel_identifier: Whitelabel ID (required)
        :type whitelabel_identifier: int
        :param async_req: Whether to execute the request asynchronously.
        :type async_req: bool, optional
        :param _return_http_data_only: response data without head status code
                                       and headers
        :type _return_http_data_only: bool, optional
        :param _preload_content: if False, the urllib3.HTTPResponse object will
                                 be returned without reading/decoding response
                                 data. Default is True.
        :type _preload_content: bool, optional
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :param _request_auth: set to override the auth_settings for an a single
                              request; this effectively ignores the authentication
                              in the spec for a single request.
        :type _request_auth: dict, optional
        :type _content_type: string, optional: force content-type for the request
        :return: Returns the result object.
                 If the method is called asynchronously,
                 returns the request thread.
        :rtype: tuple(GetWhitelabelDomainResponse, status_code(int), headers(HTTPHeaderDict))
        """

        _params = locals()

        _all_params = [
            'whitelabel_identifier'
        ]
        _all_params.extend(
            [
                'async_req',
                '_return_http_data_only',
                '_preload_content',
                '_request_timeout',
                '_request_auth',
                '_content_type',
                '_headers'
            ]
        )

        # validate the arguments
        for _key, _val in _params['kwargs'].items():
            if _key not in _all_params:
                raise ApiTypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method get_whitelabel_domain" % _key
                )
            _params[_key] = _val
        del _params['kwargs']

        _collection_formats = {}

        # process the path parameters
        _path_params = {}
        if _params['whitelabel_identifier']:
            _path_params['whitelabelIdentifier'] = _params['whitelabel_identifier']

        # process the query parameters
        _query_params = []

        # process the header parameters
        _header_params = dict(_params.get('_headers', {}))

        # process the form parameters
        _form_params = []
        _files = {}

        # process the body parameter
        _body_params = None

        # set the HTTP header `Accept`
        _header_params['Accept'] = self.api_client.select_header_accept(
            ['application/json'])  # noqa: E501

        # authentication setting
        _auth_settings = []  # noqa: E501

        _response_types_map = {
            '200': "GetWhitelabelDomainResponse",
        }

        return self.api_client.call_api(
            '/api/whitelabel/{whitelabelIdentifier}/domain', 'GET',
            _path_params,
            _query_params,
            _header_params,
            body=_body_params,
            post_params=_form_params,
            files=_files,
            response_types_map=_response_types_map,
            auth_settings=_auth_settings,
            async_req=_params.get('async_req'),
            _return_http_data_only=_params.get('_return_http_data_only'),  # noqa: E501
            _preload_content=_params.get('_preload_content', True),
            _request_timeout=_params.get('_request_timeout'),
            collection_formats=_collection_formats,
            _request_auth=_params.get('_request_auth'))

    @validate_arguments
    def update_deployment_targets(self, whitelabel_identifier : Annotated[StrictInt, Field(..., description="Whitelabel ID")], update_whitelabel_deployment_targets_request : UpdateWhitelabelDeploymentTargetsRequest, **kwargs) -> GenericApiResponse:  # noqa: E501
        """Update deployment targets

        Update some or all of the deployment targets enabled for this white label.

        :param whitelabel_identifier: Whitelabel ID (required)
        :type whitelabel_identifier: int
        :param update_whitelabel_deployment_targets_request: (required)
        :type update_whitelabel_deployment_targets_request: UpdateWhitelabelDeploymentTargetsRequest
        :param async_req: Whether to execute the request asynchronously.
        :type async_req: bool, optional
        :param _preload_content: if False, the urllib3.HTTPResponse object will
                                 be returned without reading/decoding response
                                 data. Default is True.
        :type _preload_content: bool, optional
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :return: Returns the result object.
                 If the method is called asynchronously,
                 returns the request thread.
        :rtype: GenericApiResponse
        """
        kwargs['_return_http_data_only'] = True
        return self._update_deployment_targets_with_http_info(whitelabel_identifier, update_whitelabel_deployment_targets_request, **kwargs)  # noqa: E501

    @validate_arguments
    def _update_deployment_targets_with_http_info(self, whitelabel_identifier : Annotated[StrictInt, Field(..., description="Whitelabel ID")], update_whitelabel_deployment_targets_request : UpdateWhitelabelDeploymentTargetsRequest, **kwargs):  # noqa: E501
        """Update deployment targets 

        Update some or all of the deployment targets enabled for this white label.

        :param whitelabel_identifier: Whitelabel ID (required)
        :type whitelabel_identifier: int
        :param update_whitelabel_deployment_targets_request: (required)
        :type update_whitelabel_deployment_targets_request: UpdateWhitelabelDeploymentTargetsRequest
        :param async_req: Whether to execute the request asynchronously.
        :type async_req: bool, optional
        :param _return_http_data_only: response data without head status code
                                       and headers
        :type _return_http_data_only: bool, optional
        :param _preload_content: if False, the urllib3.HTTPResponse object will
                                 be returned without reading/decoding response
                                 data. Default is True.
        :type _preload_content: bool, optional
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :param _request_auth: set to override the auth_settings for an a single
                              request; this effectively ignores the authentication
                              in the spec for a single request.
        :type _request_auth: dict, optional
        :type _content_type: string, optional: force content-type for the request
        :return: Returns the result object.
                 If the method is called asynchronously,
                 returns the request thread.
        :rtype: tuple(GenericApiResponse, status_code(int), headers(HTTPHeaderDict))
        """

        _params = locals()

        _all_params = [
            'whitelabel_identifier',
            'update_whitelabel_deployment_targets_request'
        ]
        _all_params.extend(
            [
                'async_req',
                '_return_http_data_only',
                '_preload_content',
                '_request_timeout',
                '_request_auth',
                '_content_type',
                '_headers'
            ]
        )

        # validate the arguments
        for _key, _val in _params['kwargs'].items():
            if _key not in _all_params:
                raise ApiTypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method update_deployment_targets" % _key
                )
            _params[_key] = _val
        del _params['kwargs']

        _collection_formats = {}

        # process the path parameters
        _path_params = {}
        if _params['whitelabel_identifier']:
            _path_params['whitelabelIdentifier'] = _params['whitelabel_identifier']

        # process the query parameters
        _query_params = []

        # process the header parameters
        _header_params = dict(_params.get('_headers', {}))

        # process the form parameters
        _form_params = []
        _files = {}

        # process the body parameter
        _body_params = None
        if _params['update_whitelabel_deployment_targets_request']:
            _body_params = _params['update_whitelabel_deployment_targets_request']

        # set the HTTP header `Accept`
        _header_params['Accept'] = self.api_client.select_header_accept(
            ['application/json'])  # noqa: E501

        # set the HTTP header `Content-Type`
        _content_types_list = _params.get('_content_type',
            self.api_client.select_header_content_type(
                ['application/json']))
        if _content_types_list:
                _header_params['Content-Type'] = _content_types_list

        # authentication setting
        _auth_settings = ['ApiKeyAuthentication', 'JWTAuthentication', 'JWTHttpHeaderAuthentication']  # noqa: E501

        _response_types_map = {
            '200': "GenericApiResponse",
        }

        return self.api_client.call_api(
            '/api/whitelabel/{whitelabelIdentifier}/deploymentTargets', 'POST',
            _path_params,
            _query_params,
            _header_params,
            body=_body_params,
            post_params=_form_params,
            files=_files,
            response_types_map=_response_types_map,
            auth_settings=_auth_settings,
            async_req=_params.get('async_req'),
            _return_http_data_only=_params.get('_return_http_data_only'),  # noqa: E501
            _preload_content=_params.get('_preload_content', True),
            _request_timeout=_params.get('_request_timeout'),
            collection_formats=_collection_formats,
            _request_auth=_params.get('_request_auth'))
