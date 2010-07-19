import csv

#Path to the csv document.
path = 'C:/Users/Rafael/Desktop/luminoso/luminoso/tests/CFB_Cities/Arlington.csv'
##path = 'C:/Users/Rafael/Desktop/luminoso/luminoso/tests/001-866.csv'
    
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
    temp_tags = []
    while True:
        row = csv_file.next()
        tags = []
        count = 0
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
                    return temp_tags, max_count
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

def open_csv_file(path):
    '''
    Main method. Here the csv file is opened and the helper functions are used to make sense of it.
    The code is documented every other step.
    '''
    csv_file = csv.reader(open(path, "U"))
    #First find the tags (tags should be located near the top of the ducument).
    #Documents may have titles in the 1st line.
    tags, count = find_tags(csv_file)

    #Path to the folder the study should be saved in.
    study_path = 'C:/Users/Rafael/Desktop/luminoso/luminoso/tests/CFB_Cities/Study/'
##    study_path = 'C:/Users/Rafael/Desktop/luminoso/luminoso/tests/csv_test/'

    #Store values.
    counter = 1
    #The document may contain other possible tags or key words and we need to find them.
    other_tags = []
    for row in csv_file:
        for i in range(count):
            item = row[i].split(' ')
            #Assumes possible tags are 1 word long. Adds them to the 'other_tags' list.
            if len(item) == 1 and item[0] != '' and item[0].count('.') == 0:
                row[i] = '#'+row[i]
                other_tags.extend([row[i]])
            
        #Create Files.
        create_file(study_path+'Documents/', counter, row)
        counter += 1
                
    #Create Canonical Document of tags.
    f = open(study_path+'Canonical/Tags.txt', 'w')
    #Add the Documents tags to the tag file.
    for tag in tags:
        f.write(tag+'\n')

    #Here 
    other_tags.sort()
    while len(other_tags) > 0:
        number = other_tags.count(other_tags[0])
        #If the possible tag was repeated, then I assumed it in fact is a tag.
        if number > 1:
            f.write(other_tags[0]+'\n')
            #Skip the repeated tags.
            other_tags = other_tags[number:]
        else:
            other_tags.pop(0)
    f.close()


if __name__ == '__main__':
    open_csv_file(path)
