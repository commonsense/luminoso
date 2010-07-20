import csv
import os

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
    tags = {}
    while True:
        row = csv_file.next()
        count = 0
        tags.clear()
        for item in range(len(row)):
            #Check if you got to the end of the row or if there is an empty block, which suggests
            #this is not the row that contains tags.
            if len(row[item]) == 0 or item == len(row) - 1:
                #Add the last tag if necessary.
                if row[item] != '':
                    tags['#'+clear_tag(row[item])] = []
                #If the current row has more items than the previous one, then update the max and
                #the tags.
                if max_count < count:
                    max_count = count
                    temp_tags = tags.copy()
                #If max_count wasn't surpased by the current count, we asume that we had found the
                #tags in the previous row and return it.
                elif max_count >= count and max_count > 0:
                    return temp_tags, max_count
                break
            else:
                tags['#'+clear_tag(row[item])] = []
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
    tags, tag_len = find_tags(csv_file)

    #Path to the folder the study should be saved in.
    study_path = os.path.abspath('.ssh/../Study/')
##    study_path = 'C:/Users/Rafael/Desktop/CSV_Files/CFB_Cities/Study/'
##    study_path = 'C:/Users/Rafael/Desktop/CSV_Files/csv_test/'

    #Store values.
    counter = 1
    #The document may contain other possible tags or key words and we need to find them.
    for row in csv_file:
        for i in range(tag_len):
            tags[tags.keys()[tag_len%1]].extend([row[i]])
            
        #Create Files.
        create_file(study_path+'\\Documents\\', counter, row)
        counter += 1
                
    #Create Canonical Document of tags.
    f = open(study_path+'\\Canonical\\Tags.txt', 'w')
    #Add the Documents tags to the tag file.
    for tag in tags.keys():
        
        f.write(tag.replace(' ','_')+'\n')
        #Now look at the words located bellow the tagged column and look for possible
        #canonical words.
        collumn = tags[tag]
        i = 0
        while i < len(collumn):
            if collumn[i].lower() == 'yes' or collumn[i].lower() == 'no':
                #Yes or No answer, thus we want positive and negative tags.
                f.write(tag.replace(' ','_')+'_yes\n')
                f.write('-'+tag.replace(' ','_')+'_no\n')
                break
            elif collumn.count(collumn[i]) <= 2:
                if len(collumn[i]) > 2:
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
                    f.write(tag.replace(' ','_')+'_'+collumn[i]+'\n')
                i += collumn.count(collumn[i])
                
    f.close()


if __name__ == '__main__':
    open_csv_file(file_name)
