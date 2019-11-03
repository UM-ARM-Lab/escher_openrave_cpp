inputs=(
'--p1x 0 --p1y -0.3 --p1yaw 0 --dx -1 --dy -1 --dyaw 0'
'--p1x -0.15 --p1y -0.3 --p1yaw 0 --dx -2 --dy -1 --dyaw 0'
'--p1x -0.45 --p1y -0.6 --p1yaw 0 --dx -1 --dy 0 --dyaw -1'
'--p1x -0.6 --p1y -0.75 --p1yaw -67.5 --dx -1 --dy 2 --dyaw -1'
'--p1x -0.75 --p1y -0.45 --p1yaw -90 --dx 0 --dy 2 --dyaw 0'
'--p1x -0.75 --p1y -0.15 --p1yaw -90 --dx 0 --dy 1 --dyaw 0'
'--p1x -0.75 --p1y 0 --p1yaw -90 --dx 0 --dy 2 --dyaw 0'
'--p1x -0.75 --p1y 0.3 --p1yaw -90 --dx -1 --dy 1 --dyaw 0'
'--p1x -0.9 --p1y 0.45 --p1yaw -112.5 --dx 1 --dy 2 --dyaw 1'
'--p1x -0.75 --p1y 0.75 --p1yaw -90 --dx 0 --dy 2 --dyaw 0'
'--p1x -0.75 --p1y 1.05 --p1yaw -90 --dx -1 --dy 1 --dyaw 0'
'--p1x -0.75 --p1y 1.2 --p1yaw -90 --dx -1 --dy 2 --dyaw 0'
'--p1x -0.75 --p1y 1.5 --p1yaw -90 --dx -1 --dy 1 --dyaw 0'
'--p1x -0.9 --p1y 1.65 --p1yaw -112.5 --dx 0 --dy 2 --dyaw 1'
'--p1x -0.75 --p1y 1.95 --p1yaw -90 --dx 0 --dy 2 --dyaw 0'
'--p1x -0.75 --p1y 2.25 --p1yaw -90 --dx 0 --dy 1 --dyaw 0'
'--p1x -0.75 --p1y 2.4 --p1yaw -90 --dx 0 --dy 2 --dyaw 1'
'--p1x -0.45 --p1y 2.55 --p1yaw 0 --dx 0 --dy 1 --dyaw 0'
'--p1x -0.45 --p1y 2.7 --p1yaw 0 --dx -1 --dy 1 --dyaw 0'
'--p1x 2.25 --p1y 3 --p1yaw 0 --dx -1 --dy 0 --dyaw 1'
'--p1x 7.95 --p1y 2.25 --p1yaw 22.5 --dx -1 --dy -1 --dyaw 0'
'--p1x 7.95 --p1y 1.35 --p1yaw 0 --dx 0 --dy -1 --dyaw 0'
)
for i in "${inputs[@]}"
do 
    echo ${i}
    python specific_contact_transition_sampler.py ${i}
    python specific_1.py
    python specific_2.py
    python specific_3.py
    rm specific_transitions
    rm transitions_dict_specific_transitions
    rm specific_dynamic_cost
done;
