# Import the Flask class from the flask module
from flask import Flask
from flask import request

# Create an instance of the Flask class, passing in the name of the current module
app = Flask(__name__)

# Define a route for the root URL ("/")
@app.route("/")
def hello_world():
    # Function that handles requests to the root URL
    return "Hello, World!"

@app.route("/data")
def get_data():
    try:
        if data and len(data) > 0:
            return {"message": f"Data of length {len(data)} found"}
        else:
            return {"message": "Data is empty"}, 500
    except NameError:
        # Handle the case where the data variable is not defined
        return {"message": "Data not found"}, 404
    
@app.route("/name_search")
def name_search():
    first_name = request.args.get("q")
    if first_name:
        results = [item for item in data if item["first_name"] == first_name]
        if results:
            return {"results": results}, 200
        else:
            return {"message": "Person not found"}, 404
    else:
        return {"message": "Invalid input parameter"}, 422
    
@app.route("/count")
def count():
    try:
        return {"data count": len(data)}, 200
    except NameError:
        return {"message": "Data not found"}, 404
    
@app.route("/person/<id>")
def find_by_uuid(id):
    for person in data:
        if person["id"] == str(id):
            return person, 200
        
    return {"message": "Person not found"}, 404

@app.route("/person/<id>", methods=["DELETE"])
def delete_by_uuid(id):
    for person in data:
        if person["id"] == str(id):
            data.remove(person)
            return {"message": "Person deleted"}, 200
        
    return {"message": "Person not found"}, 404

@app.route("/person", methods=["POST"])
def add_by_uuid():
    person = request.get_json()
    if not person:
        return {"message": "Invalid input"}, 400

    try:
        data.append(person)
    except NameError:
        return {"message": "Data not defined"}, 500

    return {"message": f"Person added: {person['id']}"}, 201

@app.errorhandler(404)
def api_not_found(error):
    return {"message": "Resource not found"}, 404

data = [
    {
        "id": "3b58aade-8415-49dd-88db-8d7bce14932a",
        "first_name": "Tanya",
        "last_name": "Slad",
        "graduation_year": 1996,
        "address": "043 Heath Hill",
        "city": "Dayton",
        "zip": "45426",
        "country": "United States",
        "avatar": "http://dummyimage.com/139x100.png/cc0000/ffffff",
    },
    {
        "id": "d64efd92-ca8e-40da-b234-47e6403eb167",
        "first_name": "Ferdy",
        "last_name": "Garrow",
        "graduation_year": 1970,
        "address": "10 Wayridge Terrace",
        "city": "North Little Rock",
        "zip": "72199",
        "country": "United States",
        "avatar": "http://dummyimage.com/148x100.png/dddddd/000000",
    },
    {
        "id": "66c09925-589a-43b6-9a5d-d1601cf53287",
        "first_name": "Lilla",
        "last_name": "Aupol",
        "graduation_year": 1985,
        "address": "637 Carey Pass",
        "city": "Gainesville",
        "zip": "32627",
        "country": "United States",
        "avatar": "http://dummyimage.com/174x100.png/ff4444/ffffff",
    },
    {
        "id": "0dd63e57-0b5f-44bc-94ae-5c1b4947cb49",
        "first_name": "Abdel",
        "last_name": "Duke",
        "graduation_year": 1995,
        "address": "2 Lake View Point",
        "city": "Shreveport",
        "zip": "71105",
        "country": "United States",
        "avatar": "http://dummyimage.com/145x100.png/dddddd/000000",
    },
    {
        "id": "a3d8adba-4c20-495f-b4c4-f7de8b9cfb15",
        "first_name": "Corby",
        "last_name": "Tettley",
        "graduation_year": 1984,
        "address": "90329 Amoth Drive",
        "city": "Boulder",
        "zip": "80305",
        "country": "United States",
        "avatar": "http://dummyimage.com/198x100.png/cc0000/ffffff",
    }
]

app.run()