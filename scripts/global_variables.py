def init_global_variables():
    global transition_number
    transition_number = 0
    global output_file_handler
    # write all the transition records to this file
    output_file_handle = open('records.txt', 'w')
