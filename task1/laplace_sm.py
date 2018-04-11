from collections import Counter
import string

file_name = "test_corpus.txt"
file = open(file_name, "r")
blob = file.read()
file.close()

def clean_file(file):
  table = str.maketrans('', '', string.punctuation)
  lines = blob.split("\n")
  lines = [line.lower() for line in lines] 
  return [line.translate(table) for line in lines]

def divide_emails(emails):
  ham = []
  spam = []
  for email in emails:
    if email:
      [etype, econtent] = email.split("\t")
      ham.append(econtent) if etype == "ham" else spam.append(econtent)
  return (ham, spam)

def make_word_freq(emails):
  words = []
  for email in emails:
    words.extend(email.split())
  return Counter(words)

emails = clean_file(blob)
(ham, spam) = divide_emails(emails)

if len(ham) > len(spam):
  train_ham = ham[0:int(len(spam)*0.8)]
  train_spam = spam[0:int(len(spam)*0.8)]

  cv_ham = ham[int(len(spam)*0.8):int(len(spam)*0.9)]
  cv_spam = spam[int(len(spam)*0.8):int(len(spam)*0.9)]

  test_ham = ham[int(len(spam)*0.9):len(spam)]
  test_spam = spam[int(len(spam)*0.9):len(spam)]
else:
  train_ham = ham[0:int(len(ham)*0.8)]
  train_spam = spam[0:int(len(ham)*0.8)]

  cv_ham = ham[int(len(ham)*0.8):int(len(ham)*0.9)]
  cv_spam = spam[int(len(ham)*0.8):int(len(ham)*0.9)]

  test_ham = ham[int(len(ham)*0.9):len(ham)]
  test_spam = spam[int(len(ham)*0.9):len(ham)]

p_ham = len(train_ham)/(len(train_ham)+len(train_spam))
p_spam = len(train_spam)/(len(train_ham)+len(train_spam))

ham_word_freq = make_word_freq(train_ham)
spam_word_freq = make_word_freq(train_spam)

ham_word_count = sum(ham_word_freq.values())
spam_word_count = sum(spam_word_freq.values())
vocabulary = {**ham_word_freq, **spam_word_freq}
total_word_count = len(vocabulary.keys())

def cond_word(word, spam, alpha):
  if spam:
     return (spam_word_freq[word]+alpha)/(float)(spam_word_count+alpha*total_word_count)
  return (ham_word_freq[word]+alpha)/(float)(ham_word_count+alpha*total_word_count)

def cond_email(email, spam, alpha):
  result = 1.0
  words = email.split()
  for word in words:
    result *= cond_word(word, spam, alpha)
  return result

def classify(email, alpha):
  email = email.lower()
  is_spam = p_spam*cond_email(email, True, alpha)
  is_ham = p_ham*cond_email(email, False, alpha)
  return is_spam > is_ham

results = []
for k in range(0, 100):
  result = 0
  for email in cv_ham:
    if classify(email, k):
      result-=1

  for email in cv_spam:
    if classify(email, k):
      result+=1

  results.append(result)

best_k = results.index(max(results))
print("best K", best_k)

test_fn = input("name of test document ")
result_fn = input("name of result document ")

file = open(test_fn, "r")
blob = file.read()
file.close()

file = open(result_fn, "a")

test_lns = clean_file(blob)
for line in test_lns:
  if line:
    if classify(line, best_k):
      file.write("spam\t"+line+"\n")
    else:
      file.write("ham\t"+line+"\n")

file.close()