"""Seed place names for the WeatherGPT corpora.

These are real Indian district headquarters / notable towns chosen to span every
state and union territory and every major climate zone (arid Thar, Himalayan,
Western Ghats orographic, Gangetic plain, coastal peninsular, north-east
monsoon).  Only the *names* are hardcoded; coordinates, elevation and the
administrative hierarchy are resolved from the Open-Meteo geocoding API so no
number in the dataset is a literal typed by hand.
"""

SEED_PLACES: list[str] = [
    # Maharashtra
    "Nagpur", "Mumbai", "Pune", "Nashik", "Aurangabad", "Solapur", "Amravati",
    "Kolhapur", "Ratnagiri", "Bhandara", "Yavatmal", "Osmanabad",
    # Madhya Pradesh / Chhattisgarh
    "Bhopal", "Indore", "Jabalpur", "Gwalior", "Rewa", "Raipur", "Bilaspur", "Jagdalpur",
    # Rajasthan
    "Jaipur", "Jodhpur", "Bikaner", "Jaisalmer", "Udaipur", "Kota", "Barmer", "Alwar",
    # Gujarat
    "Ahmedabad", "Rajkot", "Surat", "Bhuj", "Vadodara", "Porbandar", "Dahod",
    # Uttar Pradesh
    "Lucknow", "Kanpur", "Varanasi", "Prayagraj", "Agra", "Gorakhpur", "Bareilly", "Jhansi",
    # Bihar / Jharkhand
    "Patna", "Gaya", "Muzaffarpur", "Bhagalpur", "Ranchi", "Jamshedpur", "Daltonganj",
    # West Bengal / Sikkim
    "Kolkata", "Darjeeling", "Siliguri", "Bardhaman", "Midnapore", "Cooch Behar", "Gangtok",
    # Odisha
    "Bhubaneswar", "Cuttack", "Puri", "Sambalpur", "Koraput", "Balasore",
    # North-east
    "Guwahati", "Dibrugarh", "Silchar", "Shillong", "Imphal", "Aizawl", "Agartala",
    "Kohima", "Itanagar", "Cherrapunji", "Tezpur",
    # Punjab / Haryana / Himachal / J&K / Ladakh / Uttarakhand
    "Amritsar", "Ludhiana", "Chandigarh", "Hisar", "Rohtak", "Shimla", "Manali",
    "Dharamshala", "Srinagar", "Jammu", "Leh", "Dehradun", "Nainital", "Joshimath",
    # Delhi
    "New Delhi",
    # Telangana / Andhra Pradesh
    "Hyderabad", "Warangal", "Nizamabad", "Visakhapatnam", "Vijayawada", "Tirupati",
    "Kurnool", "Anantapur", "Kakinada",
    # Karnataka
    "Bengaluru", "Mysuru", "Mangaluru", "Hubli", "Belagavi", "Kalaburagi", "Shivamogga",
    "Chikkamagaluru", "Bidar",
    # Tamil Nadu / Puducherry
    "Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem", "Thoothukudi",
    "Nagercoil", "Ooty", "Puducherry", "Vellore",
    # Kerala
    "Thiruvananthapuram", "Kochi", "Kozhikode", "Thrissur", "Alappuzha", "Idukki", "Kannur",
    # Islands and UTs
    "Port Blair", "Kavaratti", "Daman", "Silvassa",
]
