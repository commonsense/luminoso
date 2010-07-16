import csv
path = 'C:/Users/Rafael/Desktop/luminoso/luminoso/tests/001-866.csv'
##path = 'C:/Users/Rafael/Desktop/luminoso/luminoso/tests/CFB_Cities/Arlington.csv'

f = csv.reader(open(path, "U"))

def clear_tag(str):
    '''
    Clear any unwanted formating from the tags. Remember: Tag names will be used to name the files.
    '''
    #Note: Add any other code to remove unwanted format here.
    return str.replace('\n', '').replace('?', '')

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

x = find_tags(f)

