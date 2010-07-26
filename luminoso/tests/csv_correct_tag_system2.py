import csv
import os

'''
This script allows for users to input CSV files and convert them into luminoso ready study files.
There are some premises that need to be complied for the program to run properly:
    - The user must include a Study folder in the same directory this script is saved in.
        - This folder should contain subfolders, namely Canonical, Documents, Results and
        Matrices folders inside.
    - The CSV file will probably have tags or topic names at the top (tags usually
    describe the contents of the collumn). Tags should not be repeated. This will cause
    script to give an error as it assumes each collumns has a unique tag.
'''

#Files should be in the same directory as this script. User should write file name bellow.
file_name = '001-866.csv'
##file_name = '/CFB_Cities/Arlington.csv'
    
def create_file(path, counter, stored_values):
    '''
    Takes the path to the folder to write to and writes a file containing the tag's contents.
    '''
    f = open(path + str(counter) + '.txt', 'w')
    for value in stored_values:
        f.write(value + ' ')
    f.close()

def find_tags(csv_file):
    '''
    Finds tags by looking for the 1st row with at least as many elements as the next.
    This assumes that the maximum amount of elements per row is the number of tags.
    '''
    max_count = 0
    temp_tags = dict
    tags = []
    while True:
        row = csv_file.next()
        count = 0
        tags = []
        for item in range(len(row)):
            #Check if you got to the end of the row or if there is an empty block, which suggests
            #this is not the row that contains tags.
            if len(row[item]) == 0 or item == len(row) - 1:
                #Add the last tag if necessary.
                if row[item] != '':
                    tags.extend(['#'+clear_tag(row[item])])
                #If the current row has more items than the previous one, then update the max and
                #the tags.
                if max_count < count:
                    max_count = count
                    temp_tags = tags
                #If max_count wasn't surpased by the current count, we asume that we had found the
                #tags in the previous row and return it.
                elif max_count >= count and max_count > 0:
                    dict_tags = {}.fromkeys(temp_tags)
                    for i in temp_tags:
                        dict_tags[i] = []
                    dict_tags['order'] = temp_tags
                    #The script checks one line in advance to make sure this is in fact the row
                    #containing tags and thus we need to include it.
                    skipped_row = row
                    return dict_tags, max_count, skipped_row
                break
            else:
                tags.extend(['#'+clear_tag(row[item])])
                count+=1

def clear_tag(str):
    '''
    Clear any unwanted formating from the tags. Remember: Tag names will be used to name the files.
    '''
    #Note: Add any other code to remove unwanted format here.
    return str.replace('\n', '').replace('?', '')

def open_csv_file(file_name):
    '''
    Main method. Here the csv file is opened and the helper functions are used to make sense of it.
    The code is documented every other step.
    '''
    path = os.path.abspath('.ssh/../'+file_name)
    
    csv_file = csv.reader(open(path, "U"))
    #First find the tags (tags should be located near the top of the ducument).
    #Documents may have titles in the 1st line.
    tags, tag_len, skipped_row = find_tags(csv_file)


    #Path to the folder the study should be saved in.
    study_path = os.path.abspath('.ssh/../Study/')
##    study_path = 'C:/Users/Rafael/Desktop/CSV_Files/CFB_Cities/Study/'
##    study_path = 'C:/Users/Rafael/Desktop/CSV_Files/csv_test/'
    
    #The document may contain other possible tags or key words and we need to find them.
    pointer = 0
    for x in skipped_row[:tag_len]:
        tags[tags['order'][pointer%tag_len]].extend([x])
        pointer += 1
    for row in csv_file:
        for j in row[:tag_len]:
            tags[tags['order'][pointer%tag_len]].extend([j])
            pointer += 1
            
    #List of tuples in the form (tag: {collumn_tag, edited_final_tag}). Used to replace words that
    #were chosen as tags.
    replacement_tags = {}.fromkeys(tags.keys())
    for t in replacement_tags.keys():
        replacement_tags[t] = []

    #Length of the collumns.
    len_col = len(tags[tags.keys()[0]])
    
    #Create Canonical Document of tags.
    f = open(study_path+os.sep+'Canonical'+os.sep+'Tags.txt', 'w')
    
    #tags dictionary should not be altered as it will be used later on, so we make a copy here.
    copy_tags = {}.fromkeys(tags.keys())
    for y in copy_tags.keys():
        copy_tags[y] = []
        copy_tags[y].extend(tags[y])
    
    #Add the Documents tags to the tag file.
    for tag in copy_tags.keys():
        if tag == 'order':
            continue
        f.write(tag.replace(' ','_')+'\n')
        
        #Now look at the words located bellow the tagged column and look for possible
        #canonical words.
        collumn = copy_tags[tag]
        collumn.sort()
        i = 0

        while i < len(collumn):
            if collumn[i].lower() == 'yes' or collumn[i].lower() == 'no':
                #Yes or No answer, thus we want positive and negative tags.
                yes = tag.replace(' ','_')+'_yes\n'
                no = '-'+tag.replace(' ','_')+'_no\n'

                replacement_tags[tag] = {'yes': yes}
                replacement_tags[tag]['no'] = no
                
                f.write(yes+'\n')
                f.write(no+'\n')
                break
            elif collumn.count(collumn[i]) <= 2:
                if len(collumn[i]) > 5:
                    #Assume that the collumn contains descriptions rather than possible tags.
                    break
                else:
                    #Else, maybe this is just not a tag but the collumn may contain tags.
                    i += 1
                    continue
            else:
                #If it's repeated enough then make it a tag. Always getting rid of the empty items
                #first.
                if collumn[i] != '' and collumn[i] != ' ':
                    #If we already added it, then ignore it.
                    if tag.replace(' ','_')+'_'+collumn[i].replace(' ','_') in copy_tags:
                        i += collumn.count(collumn[i])
                        continue
                    #We add new tags here so that they are not repeated.
                    copy_tags[tag.replace(' ','_')+'_'+collumn[i].replace(' ','_')] = []
                    #This is the new accepted tag.
                    new_tag = tag.replace(' ','_')+'_'+collumn[i].replace(' ','_')
                    if replacement_tags[tag] == []:
                        replacement_tags[tag] = {collumn[i] : new_tag}
                    else:
                        replacement_tags[tag][collumn[i]] = new_tag
                    #Write tags into the canonical document.
                    f.write(new_tag+'\n')
                i += collumn.count(collumn[i])
        i = 0
    
    #Store values by substracting them from the ordered dictionary we created at the beginning,
    #namely 'tags'. This extracts words from the dictionary and appends them to a list, then
    #creates a text document with the list's contents.
    counter = 0
    r = []
    while counter < len_col-1:
        for tag in tags['order']:
            item = tags[tag][counter]
            if item in replacement_tags[tag]:
                r.extend([replacement_tags[tag][item]])
            else:
                r.extend([item])
        create_file(study_path+os.sep+'Documents', counter, r)
        counter += 1
        r = []
        
    f.close()


if __name__ == '__main__':
    open_csv_file(file_name)
