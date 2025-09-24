// src/components/FeedbackForm.jsx
import React, { useState } from 'react';
import { FaStar } from 'react-icons/fa';
import { X } from 'lucide-react';
import api from '../utils/api'; // Import your configured axios instance

const FeedbackForm = ({ isOpen, onClose, onSubmit }) => {
  const [rating, setRating] = useState(0);
  const [hover, setHover] = useState(0);
  const [comment, setComment] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitSuccess, setSubmitSuccess] = useState(false);
  const [submitError, setSubmitError] = useState(null);

  if (!isOpen) return null;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    setSubmitError(null);
    setSubmitSuccess(false);

    try {
      // Send feedback to your new backend API endpoint
      const response = await api.post('/api/feedback', {
        rating: rating,
        comment: comment,
        // You might want to pass user_id or user_email from context if your backend needs it
        // e.g., userId: someUserId, userEmail: someUserEmail
      });

      console.log('Feedback submitted successfully:', response.data);
      setSubmitSuccess(true);
      setRating(0); // Reset form
      setComment(''); // Reset form
      setHover(0); // Reset form
      onSubmit({ rating, comment }); // Call the prop function if parent needs to know

      setTimeout(() => {
        onClose(); // Close after a short delay
        setSubmitSuccess(false); // Reset success state for next time
      }, 1500);

    } catch (error) {
      console.error('Error submitting feedback:', error);
      setSubmitError('Failed to submit feedback. Please try again. ' + (error.response?.data?.detail || error.message || ''));
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 backdrop-blur-sm">
      <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 p-6 rounded-xl shadow-2xl w-full max-w-md mx-4 transform transition-all duration-300 scale-100 opacity-100">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Give Feedback</h2>
          <button
            onClick={onClose}
            disabled={isSubmitting}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors duration-200"
          >
            <X size={20} className="text-gray-500 dark:text-gray-400" />
          </button>
        </div>

        {submitSuccess ? (
          <div className="text-center py-8">
            <div className="mx-auto mb-4 w-16 h-16 bg-green-100 dark:bg-green-900/20 rounded-full flex items-center justify-center">
              <svg className="w-8 h-8 text-green-600 dark:text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
              </svg>
            </div>
            <p className="text-green-600 dark:text-green-400 text-lg font-medium mb-2">Thank you for your feedback!</p>
            <p className="text-gray-600 dark:text-gray-400 text-sm">We appreciate your input and will use it to improve ReconBot.</p>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-6">
            {submitError && (
              <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-700/50 rounded-lg">
                <p className="text-red-700 dark:text-red-300 text-sm">{submitError}</p>
              </div>
            )}

            {/* Rating Section */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
                How would you rate your experience?
              </label>
              <div className="flex justify-center space-x-2 mb-2">
                {[...Array(5)].map((_, index) => {
                  const ratingValue = index + 1;
                  return (
                    <label key={index} className="cursor-pointer">
                      <input
                        type="radio"
                        name="rating"
                        value={ratingValue}
                        onClick={() => setRating(ratingValue)}
                        className="hidden"
                        disabled={isSubmitting}
                      />
                      <FaStar
                        className="transition-all duration-200 hover:scale-110"
                        color={ratingValue <= (hover || rating) ? "#ffc107" : "#e5e7eb"}
                        size={32}
                        onMouseEnter={() => !isSubmitting && setHover(ratingValue)}
                        onMouseLeave={() => !isSubmitting && setHover(0)}
                      />
                    </label>
                  );
                })}
              </div>
              {rating === 0 && (
                <p className="text-red-500 dark:text-red-400 text-xs text-center">Please select a rating</p>
              )}
            </div>

            {/* Comment Section */}
            <div>
              <label htmlFor="comment" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Comments (Optional)
              </label>
              <textarea
                id="comment"
                rows="4"
                className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400 focus:border-transparent resize-none transition-colors duration-200"
                value={comment}
                onChange={(e) => setComment(e.target.value)}
                placeholder="What can we improve? What did you like? Your feedback helps us make ReconBot better."
                disabled={isSubmitting}
              ></textarea>
            </div>

            {/* Action Buttons */}
            <div className="flex justify-end space-x-3">
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-400 dark:focus:ring-gray-500 transition-colors duration-200"
                disabled={isSubmitting}
              >
                Cancel
              </button>
              <button
                type="submit"
                className="px-6 py-2 bg-blue-600 hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200"
                disabled={isSubmitting || rating === 0}
              >
                {isSubmitting ? (
                  <div className="flex items-center">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Submitting...
                  </div>
                ) : (
                  'Submit Feedback'
                )}
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
};

export default FeedbackForm;