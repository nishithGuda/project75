import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useBankData } from "../context/BankDataContext";

const Transfer = () => {
  const { accounts, processTransfer, getAccountDetails } = useBankData();
  const navigate = useNavigate();

  const [transferForm, setTransferForm] = useState({
    fromAccount: "",
    toAccount: "",
    amount: "",
    transferDate: getCurrentDate(),
    memo: "",
  });

  const [errors, setErrors] = useState({});
  const [showConfirmation, setShowConfirmation] = useState(false);
  const [transferResult, setTransferResult] = useState(null);

  // Get current date in YYYY-MM-DD format for the date input default
  function getCurrentDate() {
    const now = new Date();
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, "0");
    const day = String(now.getDate()).padStart(2, "0");
    return `${year}-${month}-${day}`;
  }

  // Handle form field changes
  const handleChange = (e) => {
    const { name, value } = e.target;
    setTransferForm((prev) => ({
      ...prev,
      [name]: value,
    }));

    // Clear error for this field when user types
    if (errors[name]) {
      setErrors((prev) => ({
        ...prev,
        [name]: null,
      }));
    }
  };

  // Form validation
  const validateForm = () => {
    const newErrors = {};

    if (!transferForm.fromAccount) {
      newErrors.fromAccount = "Please select the source account";
    }

    if (!transferForm.toAccount) {
      newErrors.toAccount = "Please select the destination account";
    } else if (transferForm.fromAccount === transferForm.toAccount) {
      newErrors.toAccount =
        "Source and destination accounts cannot be the same";
    }

    if (
      !transferForm.amount ||
      isNaN(transferForm.amount) ||
      parseFloat(transferForm.amount) <= 0
    ) {
      newErrors.amount = "Please enter a valid amount greater than 0";
    } else {
      // Check if source account has sufficient funds
      const sourceAccount = accounts.find(
        (acc) => acc.id === parseInt(transferForm.fromAccount)
      );
      if (
        sourceAccount &&
        parseFloat(transferForm.amount) > sourceAccount.balance
      ) {
        newErrors.amount = "Insufficient funds in the selected account";
      }
    }

    if (!transferForm.transferDate) {
      newErrors.transferDate = "Please select a transfer date";
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  // Handle form submission
  const handleSubmit = (e) => {
    e.preventDefault();

    if (validateForm()) {
      // Process the transfer using our context function
      const result = processTransfer(transferForm);
      setTransferResult(result);
      setShowConfirmation(true);
    }
  };

  // Reset form and go back to transfer form
  const handleNewTransfer = () => {
    setTransferForm({
      fromAccount: "",
      toAccount: "",
      amount: "",
      transferDate: getCurrentDate(),
      memo: "",
    });
    setShowConfirmation(false);
    setTransferResult(null);
  };

  // Navigate to transactions page
  const handleViewTransactions = () => {
    navigate("/transactions");
  };

  // Get formatted account details for display
  const fromAccount = transferResult
    ? transferResult.fromAccount
    : getAccountDetails(transferForm.fromAccount);
  const toAccount = transferResult
    ? transferResult.toAccount
    : getAccountDetails(transferForm.toAccount);

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h2 className="text-2xl font-semibold text-primary mb-6">
        Transfer Money
      </h2>

      {showConfirmation ? (
        // Show confirmation screen
        <div
          data-id="transfer-confirmation"
          data-type="section"
          data-label="Transfer Confirmation"
          className="bg-white rounded-lg shadow-md p-6 border border-background"
        >
          <div className="mb-6">
            <h3 className="text-xl font-semibold text-secondary mb-2">
              Transfer Successful
            </h3>
            <p className="text-text-secondary">
              Your transfer has been scheduled successfully.
            </p>
          </div>

          <div className="bg-background rounded-lg p-4 mb-6">
            <h4 className="font-medium text-primary mb-3">Transfer Details</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-text-secondary">From</p>
                <p className="font-medium">
                  {fromAccount.type} ({fromAccount.number})
                </p>
              </div>
              <div>
                <p className="text-sm text-text-secondary">To</p>
                <p className="font-medium">
                  {toAccount.type} ({toAccount.number})
                </p>
              </div>
              <div>
                <p className="text-sm text-text-secondary">Amount</p>
                <p className="font-medium text-secondary">
                  $
                  {transferResult
                    ? transferResult.amount.toFixed(2)
                    : parseFloat(transferForm.amount).toFixed(2)}
                </p>
              </div>
              <div>
                <p className="text-sm text-text-secondary">Date</p>
                <p className="font-medium">
                  {transferResult
                    ? transferResult.date
                    : transferForm.transferDate}
                </p>
              </div>
              {(transferResult?.memo || transferForm.memo) && (
                <div className="col-span-2">
                  <p className="text-sm text-text-secondary">Memo</p>
                  <p className="font-medium">
                    {transferResult ? transferResult.memo : transferForm.memo}
                  </p>
                </div>
              )}
            </div>
          </div>

          <div className="flex justify-between">
            <button
              data-id="btn-make-another-transfer"
              data-type="button"
              data-label="Make Another Transfer"
              onClick={handleNewTransfer}
              className="px-4 py-2 bg-accent text-white rounded hover:bg-accent/80"
            >
              Make Another Transfer
            </button>

            <button
              data-id="btn-view-transactions"
              data-type="button"
              data-label="View Transactions"
              onClick={handleViewTransactions}
              className="px-4 py-2 border border-accent text-accent rounded hover:bg-accent/10"
            >
              View Transactions
            </button>
          </div>
        </div>
      ) : (
        // Show transfer form
        <div
          data-id="transfer-form-container"
          data-type="section"
          data-label="Transfer Form"
          className="bg-white rounded-lg shadow-md p-6 border border-background"
        >
          <form onSubmit={handleSubmit}>
            {/* From Account */}
            <div className="mb-6">
              <label
                className="block text-sm font-medium text-primary mb-2"
                htmlFor="fromAccount"
              >
                From Account
              </label>
              <select
                id="fromAccount"
                name="fromAccount"
                data-id="dropdown-from-account"
                data-type="dropdown"
                data-label="From Account Dropdown"
                className={`w-full border ${
                  errors.fromAccount
                    ? "border-text-primary"
                    : "border-background"
                } rounded-md p-2`}
                value={transferForm.fromAccount}
                onChange={handleChange}
              >
                <option value="" disabled>
                  Select an account
                </option>
                {accounts.map((account) => (
                  <option key={`from-${account.id}`} value={account.id}>
                    {account.type} ({account.number}) - $
                    {account.balance.toLocaleString()}
                  </option>
                ))}
              </select>
              {errors.fromAccount && (
                <p className="mt-1 text-sm text-text-primary">
                  {errors.fromAccount}
                </p>
              )}
            </div>

            {/* To Account */}
            <div className="mb-6">
              <label
                className="block text-sm font-medium text-primary mb-2"
                htmlFor="toAccount"
              >
                To Account
              </label>
              <select
                id="toAccount"
                name="toAccount"
                data-id="dropdown-to-account"
                data-type="dropdown"
                data-label="To Account Dropdown"
                className={`w-full border ${
                  errors.toAccount ? "border-text-primary" : "border-background"
                } rounded-md p-2`}
                value={transferForm.toAccount}
                onChange={handleChange}
              >
                <option value="" disabled>
                  Select an account
                </option>
                {accounts.map((account) => (
                  <option key={`to-${account.id}`} value={account.id}>
                    {account.type} ({account.number})
                  </option>
                ))}
                <option value="external">Add External Account</option>
              </select>
              {errors.toAccount && (
                <p className="mt-1 text-sm text-text-primary">
                  {errors.toAccount}
                </p>
              )}
            </div>

            {/* Amount */}
            <div className="mb-6">
              <label
                className="block text-sm font-medium text-primary mb-2"
                htmlFor="amount"
              >
                Amount
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <span className="text-text-secondary sm:text-sm">$</span>
                </div>
                <input
                  type="text"
                  id="amount"
                  name="amount"
                  data-id="input-amount"
                  data-type="input"
                  data-label="Transfer Amount Input"
                  className={`pl-7 w-full border ${
                    errors.amount ? "border-text-primary" : "border-background"
                  } rounded-md p-2`}
                  placeholder="0.00"
                  value={transferForm.amount}
                  onChange={handleChange}
                />
              </div>
              {errors.amount && (
                <p className="mt-1 text-sm text-text-primary">
                  {errors.amount}
                </p>
              )}
            </div>

            {/* Transfer Date */}
            <div className="mb-6">
              <label
                className="block text-sm font-medium text-primary mb-2"
                htmlFor="transferDate"
              >
                Transfer Date
              </label>
              <input
                type="date"
                id="transferDate"
                name="transferDate"
                data-id="input-transfer-date"
                data-type="input"
                data-label="Transfer Date Input"
                className={`w-full border ${
                  errors.transferDate
                    ? "border-text-primary"
                    : "border-background"
                } rounded-md p-2`}
                value={transferForm.transferDate}
                onChange={handleChange}
              />
              {errors.transferDate && (
                <p className="mt-1 text-sm text-text-primary">
                  {errors.transferDate}
                </p>
              )}
            </div>

            {/* Memo */}
            <div className="mb-8">
              <label
                className="block text-sm font-medium text-primary mb-2"
                htmlFor="memo"
              >
                Memo (Optional)
              </label>
              <input
                type="text"
                id="memo"
                name="memo"
                data-id="input-memo"
                data-type="input"
                data-label="Memo Input"
                className="w-full border border-background rounded-md p-2"
                placeholder="Add a note"
                value={transferForm.memo}
                onChange={handleChange}
              />
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              data-id="btn-submit-transfer"
              data-type="button"
              data-label="Transfer Money Button"
              className="w-full bg-accent hover:bg-accent/80 text-white font-medium py-2 px-4 rounded-md transition duration-200"
            >
              Transfer Money
            </button>
          </form>
        </div>
      )}

      {/* Transfer Tips */}
      <div
        data-id="transfer-tips"
        data-type="section"
        data-label="Transfer Tips"
        className="mt-6 bg-background rounded-lg p-4 border border-secondary/20"
      >
        <h3 className="text-md font-semibold text-primary mb-2">Quick Tips</h3>
        <ul className="text-sm text-secondary list-disc list-inside space-y-1">
          <li>
            Internal transfers between your accounts are processed immediately
          </li>
          <li>External transfers may take 1-3 business days to complete</li>
          <li>
            You can set up recurring transfers from the Scheduled Transfers
            section
          </li>
          <li>Daily transfer limits may apply based on your account type</li>
        </ul>
      </div>
    </div>
  );
};

export default Transfer;
