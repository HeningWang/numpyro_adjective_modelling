import json
import csv
import sys
import glob
import re


filenames = glob.glob("..\\data\\huashan*.json")

header = ""
data = []

#open output file and initialize csv writer for data
write_file = open('..\\data\\huashan.csv', 'w', newline='')
writer = csv.writer(write_file)

#open output file and initialize csv writer for participant info
write_subj_info_file = open('..\\data\\huashan_subj_info.csv', 'w', newline='')
writer_subject_info = csv.writer(write_subj_info_file)

#print(filenames);

#iterate through result files, extract info and write to outout files 
for file in filenames:
    id = re.split("huashan",(re.split("\.json", file)[0]))[1]
    #print(id);
    
    f = open(file)
    data = json.load(f)
    f.close()

    trials = data["trials"]
    subject_info = data["subject_information"]
    exp_condition = data["condition"]
    
    if trials:
        maintrials = trials[3:len(trials)]
    else:
        continue 
    if  header=="":
        #print(max(maintrials, key=len).keys());
        header = list(max(maintrials, key=len).keys())
        #header = list(maintrials[0].keys())
        header.insert(0, "id")

        header_subj_info = list(subject_info.keys())
        header_subj_info.insert(0, "time_in_minutes")
        header_subj_info.insert(0, "missed_break_three")
        header_subj_info.insert(0, "missed_break_two")
        header_subj_info.insert(0, "missed_break_one")
        header_subj_info.insert(0, "id")
        header_subj_info.insert(0, "left-right")
        
        writer.writerow(header)
        writer_subject_info.writerow(header_subj_info)
	
    #data	
    for trial in maintrials:
        list_of_data = list(trial.values())
        list_of_data.insert(0, id)
        writer.writerow(list_of_data)
    
    #subject_info
    list_of_subject_info = list(subject_info.values())
    list_of_subject_info.insert(0, data["time_in_minutes"])
    list_of_subject_info.insert(0, data["missed_break_three"])
    list_of_subject_info.insert(0, data["missed_break_two"])
    list_of_subject_info.insert(0, data["missed_break_one"])
    list_of_subject_info.insert(0, id)
    list_of_subject_info.insert(0, exp_condition)

    writer_subject_info.writerow(list_of_subject_info)

write_file.close()
write_subj_info_file.close()