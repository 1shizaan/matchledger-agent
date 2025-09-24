// src/components/Footer.jsx
import React from 'react';
import { Link } from 'react-router-dom'; // Assuming you want to use react-router-dom for internal links

const Footer = () => {
  return (
    <footer className="bg-white dark:bg-gray-800 shadow-inner mt-8 py-6 px-4 md:px-6 lg:px-8">
      <div className="container mx-auto flex flex-col md:flex-row justify-between items-center text-center md:text-left">
        {/* Copyright or Brand Info */}
        <p className="text-gray-600 dark:text-gray-400 text-sm mb-4 md:mb-0">
          &copy; {new Date().getFullYear()} MatchLedger. All rights reserved.
        </p>

        {/* Navigation Links */}
        <nav className="flex flex-wrap justify-center md:justify-end space-x-4 sm:space-x-6 text-sm">
          <Link
            to="/contact-us" // You'll need to create a ContactUs page/route
            className="text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors duration-200"
          >
            Contact Us
          </Link>
          <a
            href="https://your-guide-link.com" // Replace with your actual guide/documentation link
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors duration-200"
          >
            Guide
          </a>
          {/* The feedback link here will simply trigger the modal, not a separate page */}
          <Link
            to="/feedback" // This can be a dummy route or handled by a context/prop
            className="text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors duration-200"
          >
            Feedback
          </Link>
        </nav>
      </div>
    </footer>
  );
};

export default Footer;