#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Integration tests for BigQuery DATETIME data type support."""

import logging
import os
import secrets
import time
import unittest

import hamcrest as hc
import pytest

import apache_beam as beam
from apache_beam.io.gcp.bigquery import ReadFromBigQuery
from apache_beam.io.gcp.bigquery import WriteToBigQuery
from apache_beam.io.gcp.bigquery_tools import BigQueryWrapper
from apache_beam.io.gcp.internal.clients import bigquery
from apache_beam.io.gcp.tests.bigquery_matcher import BigqueryFullResultMatcher
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to

try:
  from apitools.base.py.exceptions import HttpError
except ImportError:
  HttpError = None

_LOGGER = logging.getLogger(__name__)


@unittest.skipIf(HttpError is None, 'GCP dependencies are not installed')
class BigQueryDatetimeIntegrationTests(unittest.TestCase):
  """Integration tests for BigQuery DATETIME data type."""

  BIG_QUERY_DATASET_ID = 'python_datetime_it_test_'

  def setUp(self):
    self.test_pipeline = TestPipeline(is_integration_test=True)
    self.runner_name = type(self.test_pipeline.runner).__name__
    self.project = self.test_pipeline.get_option('project')

    self.bigquery_client = BigQueryWrapper()
    self.dataset_id = '%s%d%s' % (
        self.BIG_QUERY_DATASET_ID, int(time.time()), secrets.token_hex(3))
    self.bigquery_client.get_or_create_dataset(self.project, self.dataset_id)
    _LOGGER.info(
        "Created dataset %s in project %s", self.dataset_id, self.project)

  def tearDown(self):
    request = bigquery.BigqueryDatasetsDeleteRequest(
        projectId=self.project, datasetId=self.dataset_id, deleteContents=True)
    try:
      _LOGGER.info(
          "Deleting dataset %s in project %s", self.dataset_id, self.project)
      self.bigquery_client.client.datasets.Delete(request)
    except HttpError:
      _LOGGER.debug(
          'Failed to clean up dataset %s in project %s',
          self.dataset_id,
          self.project)

  def create_datetime_table(self, table_name, include_repeated=False):
    """Create a table with various DATETIME field configurations."""
    table_schema = bigquery.TableSchema()

    # ID field
    id_field = bigquery.TableFieldSchema()
    id_field.name = 'id'
    id_field.type = 'INTEGER'
    id_field.mode = 'REQUIRED'
    table_schema.fields.append(id_field)

    # Required DATETIME field
    dt_required = bigquery.TableFieldSchema()
    dt_required.name = 'event_time'
    dt_required.type = 'DATETIME'
    dt_required.mode = 'REQUIRED'
    table_schema.fields.append(dt_required)

    # Nullable DATETIME field
    dt_nullable = bigquery.TableFieldSchema()
    dt_nullable.name = 'optional_time'
    dt_nullable.type = 'DATETIME'
    dt_nullable.mode = 'NULLABLE'
    table_schema.fields.append(dt_nullable)

    if include_repeated:
      # Repeated DATETIME field
      dt_repeated = bigquery.TableFieldSchema()
      dt_repeated.name = 'timestamps'
      dt_repeated.type = 'DATETIME'
      dt_repeated.mode = 'REPEATED'
      table_schema.fields.append(dt_repeated)

    table = bigquery.Table(
        tableReference=bigquery.TableReference(
            projectId=self.project,
            datasetId=self.dataset_id,
            tableId=table_name),
        schema=table_schema)
    request = bigquery.BigqueryTablesInsertRequest(
        projectId=self.project, datasetId=self.dataset_id, table=table)
    self.bigquery_client.client.tables.Insert(request)

    # Wait for table to be available
    _ = self.bigquery_client.get_table(
        self.project, self.dataset_id, table_name)

  @pytest.mark.it_postcommit
  def test_datetime_write_and_read_basic(self):
    """Test writing and reading basic DATETIME values."""
    table_name = 'datetime_basic'
    table_id = '{}.{}'.format(self.dataset_id, table_name)

    # Test data with various DATETIME formats
    input_data = [
        {
            'id': 1,
            'event_time': '2021-01-15T10:30:00',
            'optional_time': '2021-01-15T11:30:00'
        },
        {
            'id': 2,
            'event_time': '2021-06-15T00:00:00',
            'optional_time': None
        },
        {
            'id': 3,
            'event_time': '2021-12-31T23:59:59',
            'optional_time': '2000-01-01T00:00:00'
        },
        {
            'id': 4,
            'event_time': '2021-01-15T10:30:00.123456',
            'optional_time': '2021-01-15T10:30:00.654321'
        },
    ]

    table_schema = {
        "fields": [{
            "name": "id", "type": "INTEGER", "mode": "REQUIRED"
        }, {
            "name": "event_time", "type": "DATETIME", "mode": "REQUIRED"
        },
                   {
                       "name": "optional_time",
                       "type": "DATETIME",
                       "mode": "NULLABLE"
                   }]
    }

    # Write data to BigQuery
    with TestPipeline(is_integration_test=True) as p:
      _ = (
          p
          | 'CreateData' >> beam.Create(input_data)
          | 'WriteToBQ' >> WriteToBigQuery(
              table=table_id,
              schema=table_schema,
              method=WriteToBigQuery.Method.STREAMING_INSERTS,
              project=self.project))

    # Read data back and verify
    with TestPipeline(is_integration_test=True) as p:
      result = (
          p
          | 'ReadFromBQ' >> ReadFromBigQuery(
              table=table_id,
              project=self.project,
              method=ReadFromBigQuery.Method.DIRECT_READ)
          | 'ExtractDatetime' >> beam.Map(
              lambda row:
              (row['id'], row['event_time'], row['optional_time'])))

      expected_data = [
          (1, '2021-01-15T10:30:00', '2021-01-15T11:30:00'),
          (2, '2021-06-15T00:00:00', None),
          (3, '2021-12-31T23:59:59', '2000-01-01T00:00:00'),
          (4, '2021-01-15T10:30:00.123456', '2021-01-15T10:30:00.654321'),
      ]

      assert_that(result, equal_to(expected_data))

  @pytest.mark.it_postcommit
  def test_datetime_write_with_beam_rows(self):
    """Test writing DATETIME data using Beam Rows."""
    table_name = 'datetime_beam_rows'
    table_id = '{}.{}'.format(self.dataset_id, table_name)

    # Create the table first
    self.create_datetime_table(table_name)

    # Create Beam Rows with DATETIME fields
    row_elements = [
        beam.Row(
            id=1,
            event_time='2021-01-15T10:30:00',
            optional_time='2021-01-15T11:30:00'),
        beam.Row(
            id=2, event_time='2021-06-15T12:00:00', optional_time=None),
        beam.Row(
            id=3,
            event_time='2021-12-31T23:59:59',
            optional_time='2000-01-01T00:00:00')
    ]

    # Expected data for verification
    expected_data = [
        (1, '2021-01-15T10:30:00', '2021-01-15T11:30:00'),
        (2, '2021-06-15T12:00:00', None),
        (3, '2021-12-31T23:59:59', '2000-01-01T00:00:00')
    ]

    pipeline_verifiers = [
        BigqueryFullResultMatcher(
            project=self.project,
            query=(
                "SELECT id, event_time, optional_time FROM %s ORDER BY id" %
                table_id),
            data=expected_data)
    ]

    args = self.test_pipeline.get_full_options_as_args()

    with beam.Pipeline(argv=args) as p:
      _ = (
          p
          | 'CreateRows' >> beam.Create(row_elements)
          | 'ConvertToDict' >> beam.Map(
              lambda row: {
                  'id': row.id,
                  'event_time': row.event_time,
                  'optional_time': row.optional_time
              })
          | 'WriteToBQ' >> WriteToBigQuery(
              table=table_id,
              method=WriteToBigQuery.Method.STREAMING_INSERTS,
              schema={
                  "fields": [{
                      "name": "id", "type": "INTEGER", "mode": "REQUIRED"
                  },
                             {
                                 "name": "event_time",
                                 "type": "DATETIME",
                                 "mode": "REQUIRED"
                             },
                             {
                                 "name": "optional_time",
                                 "type": "DATETIME",
                                 "mode": "NULLABLE"
                             }]
              }))

    # Wait a bit for streaming inserts to complete
    time.sleep(5)

    # Verify the data was written correctly
    hc.assert_that(None, hc.all_of(*pipeline_verifiers))

  @pytest.mark.it_postcommit
  def test_datetime_repeated_fields(self):
    """Test DATETIME fields with REPEATED mode."""
    table_name = 'datetime_repeated'
    table_id = '{}.{}'.format(self.dataset_id, table_name)

    input_data = [
        {
            'id': 1,
            'event_time': '2021-01-15T10:30:00',
            'optional_time': '2021-01-15T11:30:00',
            'timestamps': [
                '2021-01-15T10:30:00',
                '2021-01-15T11:30:00',
                '2021-01-15T12:30:00'
            ]
        },
        {
            'id': 2,
            'event_time': '2021-06-15T00:00:00',
            'optional_time': None,
            'timestamps': ['2021-06-15T00:00:00', '2021-06-15T12:00:00']
        },
        {
            'id': 3,
            'event_time': '2021-12-31T23:59:59',
            'optional_time': '2000-01-01T00:00:00',
            'timestamps': []  # Empty array
        }
    ]

    table_schema = {
        "fields": [{
            "name": "id", "type": "INTEGER", "mode": "REQUIRED"
        }, {
            "name": "event_time", "type": "DATETIME", "mode": "REQUIRED"
        },
                   {
                       "name": "optional_time",
                       "type": "DATETIME",
                       "mode": "NULLABLE"
                   }, {
                       "name": "timestamps",
                       "type": "DATETIME",
                       "mode": "REPEATED"
                   }]
    }

    # Write data
    args = self.test_pipeline.get_full_options_as_args()
    with beam.Pipeline(argv=args) as p:
      _ = (
          p
          | 'CreateData' >> beam.Create(input_data)
          | 'WriteToBQ' >> WriteToBigQuery(
              table=table_id,
              schema=table_schema,
              method=WriteToBigQuery.Method.STREAMING_INSERTS))

    # Read and verify
    with beam.Pipeline(argv=args) as p:
      result = (
          p
          | 'ReadFromBQ' >> ReadFromBigQuery(
              table=table_id,
              method=ReadFromBigQuery.Method.DIRECT_READ,
              project=self.project)
          | 'ExtractData' >> beam.Map(
              lambda row: (
                  row['id'],
                  len(row['timestamps']) if row['timestamps'] else 0)))

      expected_counts = [(1, 3), (2, 2), (3, 0)]
      assert_that(result, equal_to(expected_counts))

  @pytest.mark.it_postcommit
  def test_datetime_with_microseconds(self):
    """Test DATETIME values with microsecond precision."""
    table_name = 'datetime_microseconds'
    table_id = '{}.{}'.format(self.dataset_id, table_name)

    # Test data with microsecond precision
    input_data = [
        {
            'id': 1,
            'event_time': '2021-01-15T10:30:00.000001',
            'optional_time': '2021-01-15T10:30:00.999999'
        },
        {
            'id': 2,
            'event_time': '2021-01-15T10:30:00.123456',
            'optional_time': None
        },
        {
            'id': 3,
            'event_time': '2021-01-15T10:30:00.500000',
            'optional_time': '2021-01-15T10:30:00.000000'
        }
    ]

    table_schema = {
        "fields": [{
            "name": "id", "type": "INTEGER", "mode": "REQUIRED"
        }, {
            "name": "event_time", "type": "DATETIME", "mode": "REQUIRED"
        },
                   {
                       "name": "optional_time",
                       "type": "DATETIME",
                       "mode": "NULLABLE"
                   }]
    }

    expected_data = [
        (1, '2021-01-15T10:30:00.000001', '2021-01-15T10:30:00.999999'),
        (2, '2021-01-15T10:30:00.123456', None),
        (3, '2021-01-15T10:30:00.500000', '2021-01-15T10:30:00'),
    ]

    pipeline_verifiers = [
        BigqueryFullResultMatcher(
            project=self.project,
            query=(
                "SELECT id, event_time, optional_time FROM %s ORDER BY id" %
                table_id),
            data=expected_data)
    ]

    args = self.test_pipeline.get_full_options_as_args()

    with beam.Pipeline(argv=args) as p:
      _ = (
          p
          | 'CreateData' >> beam.Create(input_data)
          | 'WriteToBQ' >> WriteToBigQuery(
              table=table_id,
              schema=table_schema,
              method=WriteToBigQuery.Method.STREAMING_INSERTS))

    hc.assert_that(p, hc.all_of(*pipeline_verifiers))

  @pytest.mark.uses_gcp_java_expansion_service
  @unittest.skipUnless(
      os.environ.get('EXPANSION_JARS'),
      "EXPANSION_JARS environment var is not provided, "
      "indicating that jars have not been built")
  def test_datetime_storage_write_api(self):
    """Test DATETIME with Storage Write API method."""
    table_name = 'datetime_storage_write'
    table_id = '{}.{}'.format(self.dataset_id, table_name)

    input_data = [{
        'id': 1,
        'event_time': '2021-01-15T10:30:00',
        'optional_time': '2021-01-15T11:30:00'
    },
                  {
                      'id': 2,
                      'event_time': '2021-06-15T12:00:00',
                      'optional_time': None
                  }]

    table_schema = {
        "fields": [{
            "name": "id", "type": "INTEGER", "mode": "REQUIRED"
        }, {
            "name": "event_time", "type": "DATETIME", "mode": "REQUIRED"
        },
                   {
                       "name": "optional_time",
                       "type": "DATETIME",
                       "mode": "NULLABLE"
                   }]
    }

    expected_data = [(1, '2021-01-15T10:30:00', '2021-01-15T11:30:00'),
                     (2, '2021-06-15T12:00:00', None)]

    pipeline_verifiers = [
        BigqueryFullResultMatcher(
            project=self.project,
            query=(
                "SELECT id, event_time, optional_time FROM %s ORDER BY id" %
                table_id),
            data=expected_data)
    ]

    args = self.test_pipeline.get_full_options_as_args()

    with beam.Pipeline(argv=args) as p:
      _ = (
          p
          | 'CreateData' >> beam.Create(input_data)
          | 'WriteToBQ' >> WriteToBigQuery(
              table=table_id,
              schema=table_schema,
              method=WriteToBigQuery.Method.STORAGE_WRITE_API))

    hc.assert_that(p, hc.all_of(*pipeline_verifiers))

  @pytest.mark.it_postcommit
  def test_datetime_file_loads_method(self):
    """Test DATETIME with FILE_LOADS method."""
    table_name = 'datetime_file_loads'
    table_id = '{}.{}'.format(self.dataset_id, table_name)

    input_data = [
        {
            'id': i,
            'event_time': f'2021-01-{i:02d}T10:30:00',
            'optional_time': (
                f'2021-01-{i:02d}T11:30:00' if i % 2 == 0 else None)
        } for i in range(1, 11)  # 10 records
    ]

    table_schema = {
        "fields": [{
            "name": "id", "type": "INTEGER", "mode": "REQUIRED"
        }, {
            "name": "event_time", "type": "DATETIME", "mode": "REQUIRED"
        },
                   {
                       "name": "optional_time",
                       "type": "DATETIME",
                       "mode": "NULLABLE"
                   }]
    }

    # Verify count and some sample data
    pipeline_verifiers = [
        BigqueryFullResultMatcher(
            project=self.project,
            query="SELECT COUNT(*) as count FROM %s" % table_id,
            data=[(10, )])
    ]

    args = self.test_pipeline.get_full_options_as_args()
    gcs_temp_location = (
        f'gs://temp-storage-for-end-to-end-tests/'
        f'bq_it_test_{int(time.time())}')

    with beam.Pipeline(argv=args) as p:
      _ = (
          p
          | 'CreateData' >> beam.Create(input_data)
          | 'WriteToBQ' >> WriteToBigQuery(
              table=table_id,
              schema=table_schema,
              method=WriteToBigQuery.Method.FILE_LOADS,
              custom_gcs_temp_location=gcs_temp_location))

    hc.assert_that(p, hc.all_of(*pipeline_verifiers))


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  unittest.main()
