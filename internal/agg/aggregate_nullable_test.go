package agg

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/pkg/v2/proto/planpb"
)

func TestGroupAggReducerNullableGroupBy(t *testing.T) {
	schema := &schemapb.CollectionSchema{
		Fields: []*schemapb.FieldSchema{
			{FieldID: 100, Name: "pk", DataType: schemapb.DataType_VarChar, IsPrimaryKey: true},
			{FieldID: 101, Name: "c1", DataType: schemapb.DataType_VarChar, Nullable: true},
			{FieldID: 102, Name: "c3", DataType: schemapb.DataType_Int32},
		},
	}

	aggregates := []*planpb.Aggregate{
		{Op: planpb.AggregateOp_sum, FieldId: 102},
	}

	reducer := NewGroupAggReducer([]int64{101}, aggregates, -1, schema)

	// Segment 1: groups A, B, NULL
	result1 := NewAggregationResult([]*schemapb.FieldData{
		{
			Type:      schemapb.DataType_VarChar,
			FieldId:   101,
			FieldName: "c1",
			Field: &schemapb.FieldData_Scalars{
				Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_StringData{
						StringData: &schemapb.StringArray{
							Data: []string{"A", "B", ""},
						},
					},
				},
			},
			ValidData: []bool{true, true, false},
		},
		{
			Type:    schemapb.DataType_Int64,
			FieldId: 102,
			Field: &schemapb.FieldData_Scalars{
				Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_LongData{
						LongData: &schemapb.LongArray{
							Data: []int64{100, 200, 50},
						},
					},
				},
			},
		},
	}, 3)

	// Segment 2: groups A, NULL, C
	result2 := NewAggregationResult([]*schemapb.FieldData{
		{
			Type:      schemapb.DataType_VarChar,
			FieldId:   101,
			FieldName: "c1",
			Field: &schemapb.FieldData_Scalars{
				Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_StringData{
						StringData: &schemapb.StringArray{
							Data: []string{"A", "", "C"},
						},
					},
				},
			},
			ValidData: []bool{true, false, true},
		},
		{
			Type:    schemapb.DataType_Int64,
			FieldId: 102,
			Field: &schemapb.FieldData_Scalars{
				Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_LongData{
						LongData: &schemapb.LongArray{
							Data: []int64{150, 75, 300},
						},
					},
				},
			},
		},
	}, 3)

	reduced, err := reducer.Reduce(context.Background(), []*AggregationResult{result1, result2})
	require.NoError(t, err)

	fieldDatas := reduced.GetFieldDatas()
	require.Len(t, fieldDatas, 2)

	groupFieldData := fieldDatas[0]
	sumFieldData := fieldDatas[1]

	groupValues := groupFieldData.GetScalars().GetStringData().GetData()
	validData := groupFieldData.GetValidData()
	sumValues := sumFieldData.GetScalars().GetLongData().GetData()

	// 4 groups: A, B, NULL, C
	require.Len(t, groupValues, 4)
	require.Len(t, validData, 4)
	require.Len(t, sumValues, 4)

	groupSums := make(map[string]int64)
	var nullSum int64
	hasNullGroup := false
	for i := 0; i < len(groupValues); i++ {
		if validData[i] {
			groupSums[groupValues[i]] = sumValues[i]
		} else {
			hasNullGroup = true
			nullSum = sumValues[i]
		}
	}

	assert.True(t, hasNullGroup, "should have a NULL group")
	assert.Equal(t, int64(250), groupSums["A"]) // 100 + 150
	assert.Equal(t, int64(200), groupSums["B"]) // 200
	assert.Equal(t, int64(300), groupSums["C"]) // 300
	assert.Equal(t, int64(125), nullSum)        // 50 + 75
}
