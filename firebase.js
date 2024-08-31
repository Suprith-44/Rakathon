import { initializeApp } from 'firebase/app';
import { getFirestore, addDoc, collection } from 'firebase/firestore'; // Import Firestore

const firebaseConfig = {
  apiKey: "AIzaSyDKIp6Tk9jvGdFGCxulwi01Go7lO3IZDFQ",
  authDomain: "rakathon-c8568.firebaseapp.com",
  projectId: "rakathon-c8568",
  storageBucket: "rakathon-c8568.appspot.com",
  messagingSenderId: "906713390835",
  appId: "1:906713390835:web:a5882546e23f4d61cde186"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const db = getFirestore(app);