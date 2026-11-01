// Who to tell about a hotspot. Office locations are approximate (city level) and
// only used to sort by distance. Check each organisation's current contact page
// before writing to them, and keep this list short and real: add an entry only
// after confirming the organisation works in that landscape.

export const CONTACTS = {
  western_ghats: [
    { name: "Karnataka Forest Department", kind: "Forest dept.", what: "Kodagu forest division, Karnataka side of the box",
      url: "https://aranya.gov.in", city: "Madikeri", lat: 12.42, lon: 75.74 },
    { name: "Kerala Forest Department", kind: "Forest dept.", what: "Wayanad divisions, Kerala side of the box",
      url: "https://forest.kerala.gov.in", city: "Kalpetta", lat: 11.61, lon: 76.08 },
    { name: "Coorg Wildlife Society", kind: "NGO", what: "Kodagu conservation society, IUCN member",
      url: "https://cws.in", city: "Madikeri", lat: 12.42, lon: 75.74 },
    { name: "Nature Conservation Foundation", kind: "NGO", what: "Western Ghats rainforest research and restoration",
      url: "https://www.ncf-india.org", city: "Mysuru", lat: 12.3, lon: 76.64 },
  ],
  odisha_mining: [
    { name: "Keonjhar Forest Division", kind: "Forest dept.", what: "Divisional forest office for the box",
      url: "https://dfokeonjhar.com", city: "Keonjhar", lat: 21.63, lon: 85.58 },
    { name: "Odisha Forest Department", kind: "Forest dept.", what: "State forest department",
      url: "https://odishaforest.in", city: "Bhubaneswar", lat: 20.3, lon: 85.82 },
    { name: "Vasundhara", kind: "NGO", what: "Forest rights and community forest governance in Odisha",
      url: "https://vasundharaodisha.org", city: "Bhubaneswar", lat: 20.3, lon: 85.82 },
  ],
  northeast_hills: [
    { name: "Mizoram Environment, Forests & Climate Change Dept.", kind: "Forest dept.", what: "State forest department",
      url: "https://www.india.gov.in/department-environment-forests-climate-change-mizoram", city: "Aizawl", lat: 23.73, lon: 92.72 },
    { name: "Aaranyak", kind: "NGO", what: "Biodiversity research and conservation across Northeast India",
      url: "https://aaranyak.org", city: "Guwahati", lat: 26.14, lon: 91.74 },
  ],
  central_india: [
    { name: "Madhya Pradesh Forest Department", kind: "Forest dept.", what: "Balaghat North and South divisions",
      url: "https://www.mpforest.gov.in", city: "Balaghat", lat: 21.81, lon: 80.18 },
    { name: "The Corbett Foundation", kind: "NGO", what: "Conservation and community work in the Kanha landscape",
      url: "https://www.corbettfoundation.org", city: "Kanha", lat: 22.28, lon: 80.62 },
  ],
};

export const EVERYWHERE = [
  { name: "Global Forest Watch", kind: "Alerts", what: "Upload the hotspot file to My GFW for weekly satellite alerts on exactly these squares",
    url: "https://www.globalforestwatch.org/my-gfw/" },
  { name: "Wildlife Trust of India", kind: "NGO", what: "National conservation NGO", url: "https://www.wti.org.in" },
  { name: "WWF-India", kind: "NGO", what: "National conservation NGO", url: "https://www.wwfindia.org" },
];

export function km(a, b) {
  const R = 6371, rad = (x) => (x * Math.PI) / 180;
  const dLat = rad(b.lat - a.lat), dLon = rad(b.lon - a.lon);
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(rad(a.lat)) * Math.cos(rad(b.lat)) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
}

export function nearest(region, spot) {
  return (CONTACTS[region] || []).map((c) => ({ ...c, km: Math.round(km(spot, c)) })).sort((a, b) => a.km - b.km);
}
