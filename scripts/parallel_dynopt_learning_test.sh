
LEARNING_RATES=("0.00005" "0.0001" "0.0005" "0.001")
LAYER_NUMS=("160" "192" "224" "256")
# LAYER_NUMS=("64" "96" "128" "160" "192" "224" "256")
# DROPOUT_RATES=("0.1" "0.2" "0.3")
DROPOUT_RATES=("0")
CONTACT_TRANSITION_CODES=("2" "3" "5")

for CONTACT_TRANSITION_CODE in ${CONTACT_TRANSITION_CODES[@]}
do
    for LEARNING_RATE in ${LEARNING_RATES[@]}
    do
        for LAYER_NUM in ${LAYER_NUMS[@]}
        do
            for DROPOUT_RATE in ${DROPOUT_RATES[@]}
            do
                echo "Run test (contact transition code = ${CONTACT_TRANSITION_CODE}) with learning rate: ${LEARNING_RATE} with layer num: ${LAYER_NUM} and drop out rate: ${DROPOUT_RATE}."
                # python dynopt_feasibility_learning.py ${LEARNING_RATE} ${LAYER_NUM} ${DROPOUT_RATE} prediction ${CONTACT_TRANSITION_CODE} &
                # python dynopt_feasibility_learning.py ${LEARNING_RATE} ${LAYER_NUM} ${DROPOUT_RATE} learning ${CONTACT_TRANSITION_CODE} &
                python dynopt_objective_learning.py ${LEARNING_RATE} ${LAYER_NUM} ${DROPOUT_RATE} learning contact_transition ${CONTACT_TRANSITION_CODE} &
            done
        done
    done
    wait
done

wait

echo "Finish the learning."