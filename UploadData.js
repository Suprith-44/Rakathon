import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';

const UploadPage = () => {
    const navigate = useNavigate();
    const [file, setFile] = useState(null);
    const [error, setError] = useState('');

    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!file) {
            setError('Please select a file.');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('http://localhost:5000/upload', {
                method: 'POST',
                body: formData,
            });

            const result = await response.json();
            if (response.ok) {
                // Handle success
                console.log(result.message);
                navigate('/dashboard');
            } else {
                setError(result.message || 'An error occurred while uploading.');
            }
        } catch (error) {
            setError('An error occurred while uploading.');
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-purple-600 to-blue-500 text-white flex flex-col justify-center items-center p-4">
            <div className="bg-white bg-opacity-20 p-8 rounded-lg shadow-lg w-full max-w-md">
                <h2 className="text-3xl font-bold mb-6 text-center">Upload Your Data</h2>
                <form onSubmit={handleSubmit} className="space-y-4">
                    <div>
                        <label className="block mb-1">Select PDF File</label>
                        <input
                            type="file"
                            accept=".pdf"
                            onChange={handleFileChange}
                            className="w-full bg-white bg-opacity-50 p-2 rounded-md"
                        />
                    </div>
                    {error && <p className="text-red-500 text-center">{error}</p>}
                    <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        className="w-full bg-purple-500 hover:bg-purple-600 text-white font-bold py-2 px-4 rounded-md transition duration-300 ease-in-out"
                        type="submit"
                    >
                        Submit
                    </motion.button>
                </form>
                <div className="mt-6 text-center">
                    <Link to="/choose-llm" className="text-purple-300 hover:underline">Back to Choose Model</Link>
                </div>
            </div>
        </div>
    );
};

export default UploadPage;
