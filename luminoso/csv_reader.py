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
        return str.replace('\n', '').replace('?', '')

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
                
        #List of tuples in the form (tag: {collumn_tag, edited_final_tag}). Used to replace words that
        #were chosen as tags.
        replacement_tags = {}.fromkeys(tags.keys())
        for t in replacement_tags.keys():
            replacement_tags[t] = []

        #Length of the collumns.
        len_col = len(tags[tags.keys()[0]])
        
        #Create Canonical Document of tags.
        f = open(self.csv_file.canonical_path()+os.sep+'Tags.txt', 'w')
        
        #Add the Documents tags to the tag file.
        for tag in tags_calc.keys():
            if tag == 'order':
                continue
            f.write(tag.replace(' ','_')+'\n')
            
            #Now look at the words located bellow the tagged column and look for possible
            #canonical words.
            collumn =tags_calc[tag]
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
                        if tag.replace(' ','_')+'_'+collumn[i].replace(' ','_') in tags_calc:
                            i += collumn.count(collumn[i])
                            continue
                        #We add new tags here so that they are not repeated.
                        tags_calc[tag.replace(' ','_')+'_'+collumn[i].replace(' ','_')] = []
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
        while counter < len_col:
            for tag in tags['order']:
                if len(tags[tag]) == 0:
                    continue
                item = tags[tag][counter]
                if item in replacement_tags[tag]:
                    r.extend([replacement_tags[tag][item]])
                else:
                    r.extend([item])
            self.create_file(counter, r)
            counter += 1
            r = []
            
        f.close()

##The script can run automatically by uncommenting the code bellow and giving it a path in place
##of the "user_inputted_path".
##if __name__ == '__main__':
##    csv_file = CSVFile(user_inputted_path)
##    reader = CSVReader(csv_file)
##    reader.read_csv()
