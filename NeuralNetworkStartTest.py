import json

# opening the JSON file
data = open('WifiData2303131230.json',)
 
print("Datatype before deserialization : "
      + str(type(data)))
    
# deserializing the data
data = json.load(data)
 
print("Datatype after deserialization : "
      + str(type(data)))