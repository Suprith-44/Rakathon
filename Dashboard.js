import React from 'react';
import { Link } from 'react-router-dom';

const DashboardPage = () => {
    return (
        <div className="min-h-screen bg-gradient-to-br from-purple-600 to-blue-500 text-white flex flex-col justify-center items-center p-4">
            <div className="bg-white bg-opacity-20 p-8 rounded-lg shadow-lg w-full max-w-md">
                <h2 className="text-3xl font-bold mb-6 text-center">Dashboard</h2>
                <div className="space-y-4">
                    <Link to="/choosellm" className="block bg-red-500 hover:bg-purple-600 text-white font-bold py-2 px-4 rounded-md text-center">
                        Create New Model
                    </Link>
                    <div className="text-center text-white">
                        <p>Your Models:</p>
                        {/* Display models here */}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default DashboardPage;
