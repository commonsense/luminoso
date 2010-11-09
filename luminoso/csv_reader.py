import csv
import os
from luminoso import study

'''
This script allows for users to input CSV files and convert them into luminoso ready study files.
'''
class CSVFile(object):
    '''
    This class contains the functions to prepare csv files for luminoso.
    Empty study directories can be created here.
    Takes name and path of a csv file as its argument.
    '''

    def __init__(self, path):
        self.path = path
        self.study_path = path.strip('.csv')+'_Study'

    def create_study_dir(self):
         #Automatically creates a study directory if it does not exist already.
        if not os.path.isdir(self.study_path):
            print 'Creating Study Directory. Path to directory is '+self.study_path
            study.StudyDirectory.make_new(self.study_path)

    def canonical_path(self):
        return self.study_path+os.sep+'Canonical'

    def matrices_path(self):
        return self.study_path+os.sep+'Matrices'

    def documents_path(self):
        return self.study_path+os.sep+'Documents'

    def results_path(self):
        return self.study_path+os.sep+'results'

class CSVReader():
    '''
    This class takes a csv file and makes sense of its contents.
    '''
    
    def __init__(self, csv_file):
        self.csv_file = csv_file

    def create_file(self, counter, stored_values):
        '''
        Takes the path to the folder to write to and writes a file containing the tag's contents.
        '''
        f = open(self.csv_file.documents_path() + os.sep + str(counter) + '.txt', 'w')
        for value in stored_values:
            f.write(value + ' ')
        f.close()

    def clear_tag(self, str):
        '''
        Clear any unwanted formating from the tags. Remember: Tag names will be used to name the files.
        '''
        #Note: Add any other code to remove unwanted format here.
        return str.replace('\n', '').replace('?', '').replace(' ','_')

    def find_tags(self, csv_file):
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
                        tags.extend(['#'+self.clear_tag(row[item])])
                        count+=1
                    #If the current row has more items than the previous one, then update the max and
                    #the tags.
                    if max_count < count:
                        max_count = count
                        temp_tags = tags
                    #If max_count wasn't surpased by the current count, we asume that we had found the
                    #tags in the previous row and return it.
                    elif max_count >= count and max_count > 0:
                        dict_tags_doc = {}
                        dict_tags_calc = {}
                        c = 0
                        for i in temp_tags:
                            if i in dict_tags_doc.keys():
                                dict_tags_doc[i+'repeated'+str(c)] = []
                                c += 1
                            else:
                                dict_tags_doc[i] = []
                            dict_tags_calc[i] = []
                        dict_tags_doc['order'] = temp_tags
                        #The script checks one line in advance to make sure this is in fact the row
                        #containing tags and thus we need to include it.
                        skipped_row = row
                        #Returns a dict with to be used to recreate txt files (dict_tags_doc) and
                        #another dict to calculate tags and other useful info (dict_tags_calc). We
                        #do this because for one order matters and the other can be disposed of.
                        return dict_tags_doc, max_count, skipped_row, dict_tags_calc
                    break
                else:
                    tags.extend(['#'+self.clear_tag(row[item])])
                    count+=1

    def read_csv(self):
        '''
        Main method. Here the csv file is opened and the helper functions are used to make sense of it.
        The code is documented every other step.
        '''

        #First make sure there is a study folder.
        self.csv_file.create_study_dir()
        
        path = self.csv_file.path
        csv_file = csv.reader(open(path, "U"))
        
        #First find the tags (tags should be located near the top of the ducument).
        #Documents may have titles in the 1st line.
        tags, tag_len, skipped_row, tags_calc = self.find_tags(csv_file)

        #Path to the folder the study should be saved in.
        study_path = self.csv_file.path
        
        #The document may contain other possible tags or key words and we need to find them.
        pointer = 0
        for x in skipped_row[:tag_len]:
            if tags['order'][pointer%tag_len].find('repeated'):
                tags_calc[tags['order'][pointer%tag_len].split('repeated')[0]].extend([x])
            else:
                tags_calc[tags['order'][pointer%tag_len]].extend([x])
            tags[tags['order'][pointer%tag_len]].extend([x])
            pointer += 1
            
        for row in csv_file:
            for j in row[:tag_len]:
                if tags['order'][pointer%tag_len].find('repeated'):
                    tags_calc[tags['order'][pointer%tag_len].split('repeated')[0]].extend([x])
                else:
                    tags_calc[tags['order'][pointer%tag_len]].extend([x])
                tags[tags['order'][pointer%tag_len]].extend([j])
                pointer += 1

        confirmed = []
        new_tags = []
        for tag in tags:
            if tag == 'order':
                continue
            for i in range(len(tags[tag])):
                total_minus_empty = len(tags[tag])-tags[tag].count('')
                item_count = tags[tag].count(tags[tag][i])
                #If item is repeated more than a fifth of the time, treat as tag!
                #Since we are substituting tags[tag][i] with its new value, we need to store the old
                #value for future reference as the previous calculation will not help us check if
                #it is a tag or not.
                if (item_count > total_minus_empty/5 and len(tags[tag][i]) > 0) or ( (tags[tag][i], tag) in confirmed):

                    if tags[tag][i] not in confirmed:
                        confirmed.extend([(tags[tag][i], tag)])
                    if tags[tag][i].lower()[len(tags[tag][i])-3:] == 'yes':
                        tags[tag][i] = '#'+tag[1:]
                    elif tags[tag][i].lower()[len(tags[tag][i])-2:] == 'no':
                        tags[tag][i] = '#-'+tag[1:]
                    else:
                        tags[tag][i] = self.clear_tag(tag+'_'+tags[tag][i])

                        
                    if tags[tag][i] not in new_tags:
                        new_tags.extend([tags[tag][i]])
                        f = open(self.csv_file.canonical_path()+os.sep+tags[tag][i]+'.txt', 'w')
                        f.write(tags[tag][i])
                        f.close()
        
        #Store values by substracting them from the ordered dictionary we created at the beginning,
        #namely 'tags'. This extracts words from the dictionary and appends them to a list, then
        #creates a text document with the list's contents.
        counter = 0
        r = []
        while counter < len(tags[tags.keys()[0]]):
            for tag in tags['order']:
                if len(tags[tag]) == 0:
                    continue
                else:
                    item = tags[tag][counter]
                    r.extend([item])
            self.create_file(counter, r)
            counter += 1
            r = []
            
##The script can run automatically by uncommenting the code bellow and giving it a path in place
##of the "user_inputted_path".
##path = 'C://Documents and Settings//Rafael//Desktop//luminoso//examples//example.csv'
##path = 'C://Documents and Settings//Rafael//Desktop//luminoso//luminoso//csv_reader//001-866.csv'
##if __name__ == '__main__':
##    csv_file = CSVFile(path)
##    reader = CSVReader(csv_file)
##    reader.read_csv()
