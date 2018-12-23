START_INDEX=$1
END_INDEX=$2
BATCH_ID=$3
MOTION_TYPES=("contact_transition" "zero_step_capture" "one_step_capture")

for MOTION_TYPE in ${MOTION_TYPES[@]}
do
    for (( INDEX=$START_INDEX;INDEX<$END_INDEX;INDEX+=100 ))
    do
        echo "python dynopt_objective_learning.py 0.0001 256 0.0 read_data ${MOTION_TYPE} 0 $INDEX $(($INDEX+100)) "
        python dynopt_objective_learning.py 0.0001 256 0.0 read_data ${MOTION_TYPE} 0 $INDEX $(($INDEX+100)) &
    done

    for (( INDEX=$START_INDEX;INDEX<$END_INDEX;INDEX+=100 ))
    do
        echo "python dynopt_feasibility_learning.py 0.0001 256 0.0 read_data ${MOTION_TYPE} 0 $INDEX $(($INDEX+100)) "
        python dynopt_feasibility_learning.py 0.0001 256 0.0 read_data ${MOTION_TYPE} 0 $INDEX $(($INDEX+100)) &
    done

    wait

    python dynopt_objective_learning.py 0.0001 256 0.0 combine_data ${MOTION_TYPE} 0 &

    python dynopt_feasibility_learning.py 0.0001 256 0.0 combine_data ${MOTION_TYPE} 0 &

    wait

    mkdir ../data/dynopt_result/${MOTION_TYPE}_dynopt_result_infeasible_$BATCH_ID
    mv ../data/dynopt_result/${MOTION_TYPE}_dynopt_result_infeasible*.txt ../data/dynopt_result/${MOTION_TYPE}_dynopt_result_infeasible_$BATCH_ID
    mv ../data/dynopt_result/${MOTION_TYPE}_dynopt_infeasible_total_data_dict_* ../data/dynopt_result/${MOTION_TYPE}_dynopt_result_infeasible_$BATCH_ID
    cp ../data/dynopt_result/${MOTION_TYPE}_dynopt_infeasible_total_data_dict ../data/dynopt_result/${MOTION_TYPE}_dynopt_result_infeasible_$BATCH_ID

    mkdir ../data/dynopt_result/${MOTION_TYPE}_dynopt_result_$BATCH_ID
    mv ../data/dynopt_result/${MOTION_TYPE}_dynopt_result*.txt ../data/dynopt_result/${MOTION_TYPE}_dynopt_result_$BATCH_ID
    mv ../data/dynopt_result/${MOTION_TYPE}_dynopt_total_data_dict_* ../data/dynopt_result/${MOTION_TYPE}_dynopt_result_$BATCH_ID
    cp ../data/dynopt_result/${MOTION_TYPE}_dynopt_total_data_dict ../data/dynopt_result/${MOTION_TYPE}_dynopt_result_$BATCH_ID

done

echo "Finish the data collecting."