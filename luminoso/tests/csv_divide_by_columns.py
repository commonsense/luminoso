import csv
##from luminoso.study import *

# path = C:/Users/Rafael/Desktop/luminoso/luminoso/tests

# 0            1                  2                3                     4                5               6
#RN, Razor PC Pref, Reasons, No Pref Why?, Trimmer Pref, Reasons, No Pref Why?

def create_file(path, tag, stored_values):
    '''
    Takes the path to the folder to write to and writes a file containing the tag's contents.
    '''
    f = open(path + tag + '.txt', 'w')
    f.write('#' + tag + '\n')
    for value in stored_values:
        f.write(value + ' ')
    f.close()

def find_tags(csv_file):
    '''
    Finds tags by looking for the 1st row with at least as many elements as the next.
    This assumes that the maximum amount of elements per row is the number of tags.
    '''
    row = csv_file.next()
    max_count = 0
    while True:
        tags = []
        count = 0
        for item in row:
            if len(item) == 0:
                if max_count < count:
                    max_count = count
                elif max_count >= count and max_count > 1:
                    return tags, max_count
                else:
                    row = csv_file.next()
                    max_count = count
                break
            else:
                tags.extend([clear_tag(item)])
                count+=1

def clear_tag(str):
    '''
    Clear any unwanted formating from the tags. Remember: Tag names will be used to name the files.
    '''
    #Note: Add any other code to remove unwanted format here.
    return str.replace('\n', '').replace('?', '')

def open_csv_file(path):
    csv_file = csv.reader(open(path, "U"))
    #First find the tags (tags should be located near the top of the ducument).
    #Documents may have titles in the 1st line.
    tags, count = find_tags(csv_file)

    #Then make a dict to store the values.
    stored_values = {}
    for tag in tags:
        stored_values[tag] = []

    #Store values.
    for row in csv_file:
        row.reverse()
        for i in range(count):
            stored_values[tags[i]].extend([row.pop()])

    #Create Canonical Document of tags.
    f = open('C:/Users/Rafael/Desktop/luminoso/luminoso/tests/csv_test/Canonical/Tags.txt', 'w')
    for tag in tags:
        f.write('#'+tag+'\n')
    f.close()

    #Create Files.
    for tag in tags:
        create_file('C:/Users/Rafael/Desktop/luminoso/luminoso/tests/csv_test/Documents/', tag, stored_values[tag])


if __name__ == '__main__':
    open_csv_file('C:/Users/Rafael/Desktop/luminoso/luminoso/tests/001-866.csv')
