__author__ = 'eduardo'

execfile("routine1.py")

from step2_lib import *

urls = [url for i, url in enumerate(corpus.urls) if labels[i] == u"D"]
crfcorpus = CRFCorpus.from_urllist(urls)

crfcorpus.to_txt("crfcorpus.txt")

bashCommand = "crfsuite tag -m CRF_3.model crfcorpus.txt"
import subprocess
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output = process.communicate()[0]

with open("tags.txt", "wb") as outputfile:
    outputfile.write(output)

crfcorpus.tag_from_file("tags.txt")

for docs in crfcorpus.documents:
    print "\n"
    print docs.url
    nes = docs.get_ne()
    for ne in nes:
        for chunk in ne[0]:
            print chunk
